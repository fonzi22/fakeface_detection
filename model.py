import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ===================== U-Net Generator =====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, use_cam=False):
        super(UNetGenerator, self).__init__()
        self.use_cam = use_cam
        
        # Adjust input channels if using CAM
        input_channels = n_channels + 1 if use_cam else n_channels
        
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x, cam=None):
        if self.use_cam and cam is not None:
            # Concatenate CAM with input
            x = torch.cat([x, cam], dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)
        return self.tanh(x)

# ===================== Discriminator with CAM support =====================
class Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.features = nn.Sequential(
            *discriminator_block(n_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        # Global Average Pooling before final classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)
        
        # Hook for CAM generation
        self.feature_maps = None
        self.features[-3].register_forward_hook(self.save_feature_maps)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def forward(self, x):
        features = self.features(x)
        pooled = self.gap(features)
        pooled = pooled.view(pooled.size(0), -1)
        validity = self.classifier(pooled)
        return validity

# ===================== CAM Generation =====================
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

# ===================== Loss Functions =====================
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class CAMGuidedLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_cam=10.0, lambda_recon=10.0):
        super(CAMGuidedLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cam = lambda_cam
        self.lambda_recon = lambda_recon
        
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, fake_pred, refined_fake_images, target_images, cam_maps):
        # Adversarial loss
        adv_loss = self.gan_loss(fake_pred, True)
        
        # CAM-based loss (focus on regions discriminator finds suspicious)
        cam_loss = torch.mean(cam_maps * torch.abs(refined_fake_images - target_images))
        
        # Reconstruction loss
        recon_loss = self.l1_loss(refined_fake_images, target_images)
        
        total_loss = (self.lambda_adv * adv_loss + 
                     self.lambda_cam * cam_loss + 
                     self.lambda_recon * recon_loss)
        
        return total_loss, adv_loss, cam_loss, recon_loss

# ===================== Training Class =====================
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

# ===================== Dataset Class =====================
class FaceDataset(Dataset):
    def __init__(self, source_paths, target_paths, transform=None):
        self.source_paths = source_paths
        self.target_paths = target_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.source_paths)
    
    def __getitem__(self, idx):
        source_image = Image.open(self.source_paths[idx]).convert('RGB')
        target_image = Image.open(self.target_paths[idx]).convert('RGB')
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return source_image, target_image

# ===================== Training Loop =====================
def train_cam_guided_gan():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Initialize models
    generator = UNetGenerator(n_channels=3, n_classes=3, use_cam=False)
    discriminator = Discriminator(n_channels=3)
    
    # Initialize trainer
    trainer = CAMGuidedGANTrainer(generator, discriminator, device)
    
    # Example training loop (you need to replace with your actual dataset)
    """
    # Create dataset and dataloader
    dataset = FaceDataset(source_paths, target_paths, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (source_images, target_images) in enumerate(dataloader):
            source_images = source_images.to(device)
            target_images = target_images.to(device)
            
            # Train step
            losses = trainer.train_step(target_images, source_images)
            
            # Print losses
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}] "
                      f"D Loss: {losses['d_loss']:.4f}, "
                      f"G Loss: {losses['g_loss']:.4f}, "
                      f"CAM Loss: {losses['cam_loss']:.4f}")
                
                # Generate and save sample images
                if i % 500 == 0:
                    initial_fake, refined_fake, cam_maps = trainer.generate_samples(source_images[:1])
                    # Save images here...
    """
    
    return trainer

# ===================== Visualization Helper =====================
def visualize_results(source_img, initial_fake, refined_fake, cam_map, real_img=None):
    """Visualize the complete pipeline results"""
    fig, axes = plt.subplots(1, 5 if real_img is not None else 4, figsize=(20, 4))
    
    # Convert tensors to numpy for visualization
    def tensor_to_numpy(tensor):
        return tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    def denormalize(tensor):
        return (tensor + 1) / 2
    
    # Source image
    axes[0].imshow(denormalize(tensor_to_numpy(source_img[0])))
    axes[0].set_title('Source Image')
    axes[0].axis('off')
    
    # Initial fake image
    axes[1].imshow(denormalize(tensor_to_numpy(initial_fake[0])))
    axes[1].set_title('Initial Fake')
    axes[1].axis('off')
    
    # CAM visualization
    axes[2].imshow(cam_map[0, 0].cpu().detach().numpy(), cmap='hot')
    axes[2].set_title('CAM (Suspicious Regions)')
    axes[2].axis('off')
    
    # Refined fake image
    axes[3].imshow(denormalize(tensor_to_numpy(refined_fake[0])))
    axes[3].set_title('Refined Fake')
    axes[3].axis('off')
    
    # Real image (if provided)
    if real_img is not None:
        axes[4].imshow(denormalize(tensor_to_numpy(real_img[0])))
        axes[4].set_title('Real Target')
        axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize and train the model
    trainer = train_cam_guided_gan()
    print("CAM-Guided GAN initialized successfully!")
    print("Replace the training loop with your actual dataset and start training.")