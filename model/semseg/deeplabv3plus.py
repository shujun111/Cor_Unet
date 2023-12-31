import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()
        self.is_corr = True

        if 'resnet' in cfg['backbone']:
            self.backbone = \
                resnet.__dict__[cfg['backbone']](cfg['pretrain'], multi_grid=cfg['multi_grid'],
                                                 replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Sequential(nn.Conv2d(256, cfg['nclass'], 1, bias=True)
                                        )

        if self.is_corr:
            self.corr = Corr(nclass=cfg['nclass'])
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x, need_fp=False, use_corr=True):
        # x.shape = torch.Size([2, 3, 352, 352])
        dict_return = {}
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        # c1 =torch.Size([4, 256, 88, 88])
        # c4 = torch.Size([4, 2048, 44, 44])
        c1, c4 = feats[0], feats[-1]

        if need_fp:#True 
            # feats_decode torch.Size([8, 256, 88, 88])
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            # outs torch.Size([8, 2, 88, 88])
            outs = self.classifier(feats_decode)
            # outs = torch.Size([8, 2, 88, 88])
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            # outs = torch.Size([8, 2, 352, 352])
            out, out_fp = outs.chunk(2)
            # out 4 2 352 352
            if use_corr:
                # proj_feats 4 256 44 44
                proj_feats = self.proj(c4)
                corr_out = self.corr(proj_feats, out)
                # corr_out 4 2 352 352
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        # 功能：利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if use_corr:    # True
            proj_feats = self.proj(c4)
            # 计算
            corr_out = self.corr(proj_feats, out)
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return

    def _decode(self, c1, c4):# after
        # c4 torch.Size([2, 2048, 44, 44])
        # c1 torch.Size([2, 256, 88, 88])
        c4 = self.head(c4)
        # c4 torch.Size([2, 256, 44, 44])
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        # c4 torch.Size([2, 256, 88, 88])
        c1 = self.reduce(c1)
        # c1 torch.Size([2, 48, 88, 88])
        feature = torch.cat([c1, c4], dim=1)
        # feature torch.Size([2, 304, 88, 88])
        feature = self.fuse(feature)
        # feature torch.Size([2, 256, 88, 88])

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

# correlation Map
class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        return out

