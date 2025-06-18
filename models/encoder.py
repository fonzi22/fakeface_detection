import torch
import torch.nn as nn
import torchvision.models as models

class FaceDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # retain conv layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 1)  # binary real/fake logits
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = self.pool(feat)
        logits = self.head(pooled)
        return logits.squeeze(1), feat  # return conv feat for CAM