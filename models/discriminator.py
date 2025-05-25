import torch.nn as nn
from cross_attention import AttributeAwareCrossAttention
from encoder import Encoder
    
# Discriminator (for adversarial training)
class Discriminator(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Discriminator, self).__init__()
        self.face_encoder = Encoder(embedding_dim=embedding_dim)
        self.attr_encoder = Encoder(embedding_dim=embedding_dim)
        self.cross_attention = AttributeAwareCrossAttention(embedding_dim, embedding_dim, embedding_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        face_features = self.face_encoder(x)
        attr_features = self.attr_encoder(x)
        cross_attn_features = self.cross_attention(face_features, attr_features)
        gap_features = self.gap(cross_attn_features)
        gap_features = gap_features.view(gap_features.size(0), -1)
        logits = self.classifier(gap_features)
        return logits
