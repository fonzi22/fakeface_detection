import torch
import torch.nn as nn
import torchvision.models as models

class AttributeHeader(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.eyes = nn.Linear(in_channels, 4)
        self.nose = nn.Linear(in_channels, 4)
        self.mouth = nn.Linear(in_channels, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        feat = self.pool(x)
        feat = feat.view(feat.size(0), -1)
        eyes_logits = self.eyes(feat)
        nose_logits = self.nose(feat)
        mouth_logits = self.mouth(feat)
        logits = torch.cat((eyes_logits, nose_logits, mouth_logits), dim=1)
        logits = logits.view(logits.size(0), 3, 4)
        return logits