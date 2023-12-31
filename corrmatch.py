import argparse
import logging
import os
import pprint

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
import sys
sys.path.append('/home/ljs/code/CorrMatch-main')
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.unet_corrmatch import UNet
from evaluate import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.thresh_helper import ThreshController
import random


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', default='/home/ljs/code/CorrMatch-main1/configs/pascal.yaml', type=str)
parser.add_argument('--labeled-id-path',default='/home/ljs/dataset/CH_Train_nb/img.txt', type=str)
parser.add_argument('--unlabeled-id-path',default='/home/ljs/dataset/CH_Train_nb/img.txt', type=str)
parser.add_argument('--save-path', default='/home/ljs/code/CorrMatch-main/result/Unet', type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

# def criterion_l(inputs, target, ignore_index: int = -100):
#     loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index)
#     return loss
def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()
    # 载入设置
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, word_size = setup_distributed(port=args.port)
    rank = 0
    
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0, False)
    # -------------模型定义--------------
    # model = DeepLabV3Plus(cfg)
    model = UNet(in_channels=3, num_classes=2, base_c=32)
    # 计算参数数量
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    # 优化器设置
    # optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
    #                  {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
    #                   'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    # optimizer = SGD([{'params': [param for name, param in model.named_parameters() ],
    #                   'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    optimizer = SGD([{'params': [param for name, param in model.named_parameters() ],
                      'lr': cfg['lr'] }], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    # 分布式训练设置
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=False)
    # 定义损失函数
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
        # criterion_l = nn.functional.cross_entropy(ignore_index=-100)
        
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    # Luh
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    # Lus
    criterion_kl = nn.KLDivLoss(reduction='none').cuda()
    # ------------------数据导入------------------
    # unlabeled
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    # labeled
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=True, sampler=None)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    thresh_controller = ThreshController(nclass=2, momentum=0.999, thresh_init=cfg['thresh_init'])
    
    # 开始训练
    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_loss_corr_ce, total_loss_corr_u = 0.0, 0.0
        total_mask_ratio = 0.0

        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))
        # 对于不同需求提取返回数据
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
                (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):
            # 标签数据
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            # 未标注数据
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            # 验证？
            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix)['out'].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            model.train()
            
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            # 模型的两个输入？
            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            # preds torch.Size([4, 2, 352, 352])
            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            # correlation Map---------
            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            # 侧枝
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.detach()
            # M
            conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.detach().argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            # loss_x
            # pred_x torch.Size([2, 2, 352, 352])
            # mask_x---------------------------------LOSS
            # 其中pred_x有负值
            loss_x = criterion_l(pred_x, mask_x)
            # pred_x_corr torch.Size([2, 2, 352, 352])
            # mask_x torch.Size([2, 352, 352])
            # --------------------------------------
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w
            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != -1).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)

            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa, dim=1) * conf_fliter_u_w
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()
            # -------------loss----------------
            # loss_x
            # loss_x_corr
            # loss_u_s1
            # loss_u_kl
            # loss_u_w_fp
            # loss_u_corr
            # loss = 0.5(Ls+Lu)
            # Ls = 0.5(Lsh+Lsc)
            # Lu = 0.5Luh+0.25Lus+0.25Luc
            loss = (0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += loss_u_s1.item()
            total_loss_kl += loss_u_kl.item()
            total_loss_w_fp += loss_u_w_fp.item()
            total_loss_corr_ce += loss_x_corr.item()
            total_loss_corr_u += loss_u_corr.item()
            total_mask_ratio += ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[0]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                tbar.set_description(' Total loss: {:.3f}, Loss x: {:.3f}, loss_corr_ce: {:.3f} '
                                     'Loss s: {:.3f}, Loss w_fp: {:.3f},  Mask: {:.3f}, loss_u_corr: {:.3f}'.format(
                    total_loss / (i + 1), total_loss_x / (i + 1), total_loss_corr_ce / (i + 1), total_loss_s / (i + 1),
                    total_loss_w_fp / (i + 1), total_mask_ratio / (i + 1), total_loss_corr_u / (i + 1)))
                tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        res_val = evaluate(model, valloader, eval_mode, cfg)
        mIOU = res_val['mIOU']
        class_IOU = res_val['iou_class']
        # torch.distributed.barrier()

        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.4f} \n'.format(eval_mode, mIOU))
            logger.info('***** ClassIOU ***** >>>> \n{}\n'.format(class_IOU))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            # torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
            torch.save(model.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
        # torch.distributed.barrier()


if __name__ == '__main__':
    main()
