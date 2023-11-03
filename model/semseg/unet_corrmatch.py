from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.is_corr = True
        
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)       # 64 
        self.down2 = Down(base_c * 2, base_c * 4)   # 128
        self.down3 = Down(base_c * 4, base_c * 8)   # 256
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)# 512
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.classifier = nn.Conv2d(256, 3, 1, bias=True)
                                        
        if self.is_corr:
            self.corr = Corr(nclass=2)
            self.proj = nn.Sequential(
                # 512 32
                nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x: torch.Tensor,  need_fp=False, use_corr=False) -> Dict[str, torch.Tensor]:
        dict_return = {}
        h, w = x.shape[-2:]
        
        # encoder
        x1 = self.in_conv(x)    # C1 32 352
        x2 = self.down1(x1)     # 64    176
        x3 = self.down2(x2)     # 128   88
        x4 = self.down3(x3)     # 256   44
        x5 = self.down4(x4)     # C5 256    22
        # decoder
        x = self.up1(x5, x4)    # 44
        x = self.up2(x, x3)     # 88
        x = self.up3(x, x2)     # 176
        x = self.up4(x, x1)     # 352
        
        if need_fp:
            # x torch.Size([4, 32, 352, 352])
            logits = self.out_conv(torch.cat((x, nn.Dropout2d(0.5)(x))))
            
            out, out_fp = logits.chunk(2)
            if use_corr:
                proj_feats = self.proj(x4)
                # out = [4 2 352 352]
                # proj_feats[4 32 22 22]
                corr_out = self.corr(proj_feats, out)
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return

        out = self.out_conv(x)
        if use_corr:    # True
            proj_feats = self.proj(x4)
            # 计算
            corr_out = self.corr(proj_feats, out)
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return

        
        # return {"out": logits}
    
    # def _decode(self, c1, c4):
    #     c4 = self.head(c4)
    #     c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

    #     c1 = self.reduce(c1)

    #     feature = torch.cat([c1, c4], dim=1)
    #     feature = self.fuse(feature)

    #     return feature
    
# correlation Map
class Corr(nn.Module):
    def __init__(self, nclass=2):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(32, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(32, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        # in torch.Size([4, 32, 22, 22])
        # out = [4 2 352 352]
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        # out_temp 2 2 484
        # corr_map 4 484 484
        out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        # out torch.Size([4, 2, 22, 22])
        return out