import torch

def compute_cam(model, x, class_idx=1):
    with torch.no_grad():
        logits, feat = model(x)
        # feat: [B, C, H, W], weight: [2, C]
        weight = model.fc.weight[class_idx]  # [C]
        cam = torch.einsum('bcxy,c->bxy', feat, weight)
        cam -= cam.amin(dim=[1,2], keepdim=True)
        cam /= cam.amax(dim=[1,2], keepdim=True).clamp_min(1e-6)
    return cam.unsqueeze(1)
