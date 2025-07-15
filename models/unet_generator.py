import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CamAdapter(nn.Module):
    def __init__(self, out_ch: int):
        super().__init__()
        self.adapter = nn.Conv2d(1, out_ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, cam: torch.Tensor, target_spatial: Tuple[int,int]) -> torch.Tensor:
        # cam: (B,1,Hc,Wc), target_spatial e.g. (Hf,Wf)
        # 1) resize
        cam_up = F.interpolate(cam, size=target_spatial, mode='bilinear', align_corners=False)
        # 2) project
        return self.adapter(cam_up)


class UNetGenerator(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1

        # -- encoder --
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024//factor)

        # -- CAM adapters (fixed!) --
        self.cam5 = CamAdapter(1024//factor)  # 512 if factor=2
        self.cam4 = CamAdapter(512)           # was 512//factor
        self.cam3 = CamAdapter(256)           # was 256//factor
        self.cam2 = CamAdapter(128)           # was 128//factor
        self.cam1 = CamAdapter(64)            # unchanged

        # -- decoder --
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, cam):
        # encode
        x1 = self.inc(x)       # [B,  64,  H,   W]
        x2 = self.down1(x1)    # [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # [B, 512, H/8, W/8]
        x5 = self.down4(x4)    # [B,512,  H/16,W/16]

        # CAM modulation (now shape-matched)
        x5 = x5 + x5 * torch.sigmoid(self.cam5(cam, target_spatial=x5.shape[2:]))
        
        # decode
        x = self.up1(x5, x4 + x4 * torch.sigmoid(self.cam4(cam, target_spatial=x4.shape[2:])))
        x = self.up2(x, x3 + x3 * torch.sigmoid(self.cam3(cam, target_spatial=x3.shape[2:])))
        x = self.up3(x, x2 + x2 * torch.sigmoid(self.cam2(cam, target_spatial=x2.shape[2:])))
        x = self.up4(x, x1 + x1 * torch.sigmoid(self.cam1(cam, target_spatial=x1.shape[2:])))

        logits = self.outc(x)
        return torch.tanh(logits)

