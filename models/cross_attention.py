import torch.nn as nn
import torch
import torch.nn.functional as F

class AttributeAwareCrossAttention(nn.Module):
    def __init__(self, in_channels, attr_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(attr_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(attr_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attr):
        # x: [B, C, H, W], attr: [B, C_attr, H, W]
        q = self.query_conv(x)  # [B, C_out, H, W]
        k = self.key_conv(attr) # [B, C_out, H, W]
        v = self.value_conv(attr) # [B, C_out, H, W]

        B, C, H, W = q.shape
        q_flat = q.view(B, C, -1)          # [B, C, HW]
        k_flat = k.view(B, C, -1)          # [B, C, HW]
        v_flat = v.view(B, C, -1)          # [B, C, HW]

        attn = torch.bmm(q_flat.transpose(1,2), k_flat)  # [B, HW, HW]
        attn = self.softmax(attn)                        # [B, HW, HW]

        out = torch.bmm(v_flat, attn.transpose(1,2))     # [B, C, HW]
        out = out.view(B, C, H, W)                       # [B, C, H, W]
        out = out + x                                    # Residual connection

        return out