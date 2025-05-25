import torch
import torch.nn as nn
import torch.optim as optim
from losses import GANLoss, CAMGuidedLoss
from models import UNetGenerator, Discriminator
from cam_generator import CAMGenerator

class CAMGuidedGANTrainer:
    def __init__(self, generator, discriminator, device='cuda'):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Initialize CAM generator
        self.cam_generator = CAMGenerator(self.discriminator)
        
        # Initialize refined generator (uses CAM)
        self.refined_generator = UNetGenerator(n_channels=3, n_classes=3, use_cam=True).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.refined_g_optimizer = optim.Adam(self.refined_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        self.gan_loss = GANLoss()
        self.cam_guided_loss = CAMGuidedLoss()
    
    def train_step(self, real_images, source_images):
        batch_size = real_images.size(0)
        
        # =================== Train Discriminator ===================
        self.d_optimizer.zero_grad()
        
        # Real images
        real_pred = self.discriminator(real_images)
        d_real_loss = self.gan_loss(real_pred, True)
        
        # Generate initial fake images
        with torch.no_grad():
            initial_fake_images = self.generator(source_images)
        
        # Generate CAM maps
        cam_maps = self.cam_generator.generate_cam(initial_fake_images)
        
        # Generate refined fake images
        refined_fake_images = self.refined_generator(source_images, cam_maps)
        
        # Discriminator loss on refined fake images
        fake_pred = self.discriminator(refined_fake_images.detach())
        d_fake_loss = self.gan_loss(fake_pred, False)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        self.d_optimizer.step()
        
        # =================== Train Generator ===================
        self.g_optimizer.zero_grad()
        self.refined_g_optimizer.zero_grad()
        
        # Generate initial fake images
        initial_fake_images = self.generator(source_images)
        
        # Generate CAM maps (with gradients)
        fake_pred_for_cam = self.discriminator(initial_fake_images)
        cam_maps = self.cam_generator.generate_cam(initial_fake_images)
        
        # Generate refined fake images
        refined_fake_images = self.refined_generator(source_images, cam_maps)
        
        # Get discriminator prediction on refined fake images
        refined_fake_pred = self.discriminator(refined_fake_images)
        
        # Calculate generator losses
        g_loss, adv_loss, cam_loss, recon_loss = self.cam_guided_loss(
            refined_fake_pred, refined_fake_images, real_images, cam_maps
        )
        
        # Initial generator loss (to improve initial generation)
        initial_g_loss = self.gan_loss(fake_pred_for_cam, True)
        
        total_g_loss = g_loss + 0.5 * initial_g_loss
        total_g_loss.backward()
        
        self.g_optimizer.step()
        self.refined_g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'adv_loss': adv_loss.item(),
            'cam_loss': cam_loss.item(),
            'recon_loss': recon_loss.item(),
            'initial_g_loss': initial_g_loss.item()
        }
    
    def generate_samples(self, source_images, save_cam=True):
        """Generate samples showing the improvement process"""
        self.generator.eval()
        self.refined_generator.eval()
        
        with torch.no_grad():
            # Generate initial fake images
            initial_fake = self.generator(source_images)
            
            # Generate CAM maps
            cam_maps = self.cam_generator.generate_cam(initial_fake)
            
            # Generate refined fake images
            refined_fake = self.refined_generator(source_images, cam_maps)
        
        if save_cam:
            return initial_fake, refined_fake, cam_maps
        else:
            return initial_fake, refined_fake
