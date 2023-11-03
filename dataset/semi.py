# from dataset.transform import *
import sys
sys.path.append('/home/ljs/code/CorrMatch-main')
from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import argparse


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        # self.root = root
        self.root = '/home/ljs/dataset/CH_Train'
        self.mini = '/home/ljs/dataset/Mini'
        # self.root_tmask = '/home/ljs/dataset/CH_Train'
        self.mode = mode
        # crop size
        self.size = size
        # self.complex_data_augmentation = strong_img_aug()
        self.val = '/home/ljs/dataset/CH_Test'
        mini_path = '/home/ljs/dataset/Mini/mini.txt'
        val_path = '/home/ljs/dataset/CH_Test/Test.txt'
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
            # with open(mini_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            # with open('partitions/%s/val.txt' % name, 'r') as f:
            # with open('/home/ljs/dataset/Mini/mini.txt', 'r') as f:
            with open('/home/ljs/dataset/CH_Test/Test.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        # CHL8
        if self.mode == 'train_l' or self.mode == 'train_u':
            # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            img = Image.open(os.path.join(self.root, 'img', id)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, 'mask', id )))/255)
        else:
            img = Image.open(os.path.join(self.val, 'img', id)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.val, 'mask', id )))/255)
        # Mini
        # if self.mode == 'train_l' or self.mode == 'train_u':
        #     # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        #     img = Image.open(os.path.join(self.mini, 'img', id)).convert('RGB')
        #     mask = Image.fromarray(np.array(Image.open(os.path.join(self.mini, 'mask', id )))/255)
        # else:
        #     img = Image.open(os.path.join(self.mini, 'img', id)).convert('RGB')
        #     mask = Image.fromarray(np.array(Image.open(os.path.join(self.mini, 'mask', id )))/255)
        # if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        
        # img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = -100
        # ignore_value = 254 if self.mode == 'train_u' else 255
        # img, mask = crop(img, mask, self.size, ignore_value)
        # img, mask = hflip(img, mask, p=0.5)
        # 'patch_115_6_by_5_LC08_L1TP_148033_20210415_20210424_01_T1.png'
        if self.mode == 'train_l':
            return normalize(img, mask)
        # 弱增强/强增强
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        # img_s1 = self.complex_data_augmentation(img_s1)
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        # img_s2 = self.complex_data_augmentation(img_s2)
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return len(self.ids)
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', default='/home/ljs/code/CorrMatch-main/configs/pascal.yaml', type=str)
    parser.add_argument('--labeled-id-path',default='/home/ljs/dataset/CH_Train_nb/img.txt', type=str)
    parser.add_argument('--unlabeled-id-path',default='/home/ljs/dataset/CH_Train_nb/img.txt', type=str)
    parser.add_argument('--save-path', default='/home/ljs/code/CorrMatch-main/result/Unet', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                cfg['crop_size'], args.unlabeled_id_path)
# 在模式为 'val' 时：

# img: 经过归一化处理后的图像数据
# mask: 经过归一化处理后的掩模数据
# id: 样本的 ID
# 在模式为 'train_l' 时：

# 返回一个包含两元素的元组：
# 归一化处理后的图像数据
# 归一化处理后的掩模数据
# 在其他模式下：

# 返回一个包含多个元素的元组，依次为：
# 归一化处理后的原始图像数据 img_w
# 经过数据增强处理后的第一个图像数据 img_s1
# 经过数据增强处理后的第二个图像数据 img_s2
# 归一化处理后的忽略掩模数据 ignore_mask
# 剪切框1的信息 cutmix_box1
# 剪切框2的信息 cutmix_box2
# 每个元素的含义如下：

# img, mask, img_w, img_s1, img_s2: 图像数据
# ignore_mask: 忽略掩模数据
# id: 样本的 ID
# cutmix_box1, cutmix_box2: 表示用于 CutMix 数据增强的框的信息
