import torch
import torch.nn.functional as F
from torch import nn

class CAMGenerator:
    def __init__(self, discriminator):
        self.discriminator = discriminator
    
    def generate_cam(self, images):
        """Generate Class Activation Maps for given images"""
        self.discriminator.eval()
        
        with torch.no_grad():
            # Forward pass to get predictions and feature maps
            predictions = self.discriminator(images)
            feature_maps = self.discriminator.feature_maps
            
            # Get classifier weights
            classifier_weights = self.discriminator.classifier.weight.data
            
            batch_size, num_channels, h, w = feature_maps.shape
            cam_maps = torch.zeros((batch_size, 1, images.shape[2], images.shape[3]))
            
            for i in range(batch_size):
                # Generate CAM for fake class (assuming fake = 0, real = 1)
                cam = torch.zeros((h, w))
                
                for j in range(num_channels):
                    cam += classifier_weights[0, j] * feature_maps[i, j, :, :]
                
                # Apply ReLU and normalize
                cam = F.relu(cam)
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Resize to original image size
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                                  size=(images.shape[2], images.shape[3]), 
                                  mode='bilinear', align_corners=False)
                cam_maps[i] = cam.squeeze(0)
            
            return cam_maps.to(images.device)