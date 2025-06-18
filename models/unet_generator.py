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
    """Adapter to inject CAM via 1x1 conv."""
    def __init__(self, out_ch: int):
        super().__init__()
        self.adapter = nn.Conv2d(1, out_ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, cam: Optional[torch.Tensor]) -> torch.Tensor:
        return self.adapter(cam) if cam is not None else 0.0

class UNetGenerator(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNetGenerator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cam_adapter = CamAdapter(64) 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, cam):
        x1 = self.inc(x) + self.cam_adapter(cam)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return  torch.tanh(logits)

# class DoubleConv(nn.Module):
#     """(Conv -> BN -> ReLU) x 2 helper block."""
#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)

# class DownBlock(nn.Module):
#     """Downsampling block: DoubleConv + MaxPool."""
#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.double_conv = DoubleConv(in_ch, out_ch)
#         self.pool = nn.MaxPool2d(kernel_size=2)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         features = self.double_conv(x)
#         pooled = self.pool(features)
#         return features, pooled

# class UpBlock(nn.Module):
#     """Upsampling block: Upsample/ConvTranspose + DoubleConv."""
#     def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         # Handle size mismatch due to possible odd input size
#         diffY = x2.size(2) - x1.size(2)
#         diffX = x2.size(3) - x1.size(3)
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class CamAdapter(nn.Module):
#     """Adapter to inject CAM via 1x1 conv."""
#     def __init__(self, out_ch: int):
#         super().__init__()
#         self.adapter = nn.Conv2d(1, out_ch, kernel_size=1, bias=False)
#         nn.init.zeros_(self.adapter.weight)

#     def forward(self, cam: Optional[torch.Tensor]) -> torch.Tensor:
#         return self.adapter(cam) if cam is not None else 0.0

# class UNetGenerator(nn.Module):
#     def __init__(self, in_ch: int = 3, base: int = 64):
#         super().__init__()
#         self.cam_adapter = CamAdapter(base)

#         self.inc = DoubleConv(in_ch, base)
#         self.down1 = DownBlock(base, base * 2)
#         self.down2 = DownBlock(base * 2, base * 4)
#         self.down3 = DownBlock(base * 4, base * 8)
#         self.down4 = DownBlock(base * 8, base * 8)

#         self.up1 = UpBlock(base * 16, base * 4)
#         self.up2 = UpBlock(base * 8, base * 2)
#         self.up3 = UpBlock(base * 4, base)
#         self.up4 = UpBlock(base * 2, base)

#         self.outc = nn.Conv2d(base, 3, kernel_size=1)

#     def forward(self, x: torch.Tensor, cam: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # Inject CAM at first layer
#         x0 = self.inc(x) + self.cam_adapter(cam)
#         print(x0.shape)
#         f1, p1 = self.down1(x0)
#         print(f1.shape, p1.shape)
#         f2, p2 = self.down2(p1)
#         print(f2.shape, p2.shape)
#         f3, p3 = self.down3(p2)
#         print(f3.shape, p3.shape)
#         f4, p4 = self.down4(p3)
#         print(f4.shape, p4.shape)

#         u1 = self.up1(p4, f4)
#         print(u1.shape, f3.shape)
#         u2 = self.up2(u1, f3)
#         print(u2.shape, f2.shape)
#         u3 = self.up3(u2, f2)
#         print(u3.shape, f1.shape)
#         u4 = self.up4(u3, f1)
#         print(u4.shape, x0.shape)
#         out = torch.tanh(self.outc(u4))  # output in [-1, 1]
#         return out
