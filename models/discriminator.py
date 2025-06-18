import torch
import torch.nn as nn
import torchvision.models as models

class FaceDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # retain conv layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(backbone.fc.in_features, 2)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = self.pool(feat)
        logits = self.fc(pooled.view(pooled.size(0), -1))  # flatten for fc layer
        return logits, feat  # return conv feat for CAM