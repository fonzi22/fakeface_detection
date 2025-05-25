import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import StepLR

# Định nghĩa các thông số cấu hình
class Config:
    def __init__(self):
        self.batch_size = 8
        self.lr = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epochs = 200
        self.img_size = 256
        self.channels = 3
        self.lambda_adv = 1.0
        self.lambda_rec = 10.0
        self.lambda_percep = 1.0
        self.lambda_cam = 5.0
        self.checkpoint_interval = 5
        self.sample_interval = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_dir = "results"
        self.checkpoint_dir = "checkpoints"

# Tạo thư mục cho kết quả và checkpoints
def create_dirs(config):
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

# --------------- ĐỊNH NGHĨA CÁC MÔ HÌNH ---------------

# 1. Định nghĩa khối Conditional U-Net (Generator)
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
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Xử lý kích thước nếu không khớp
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, with_feedback=True):
        super(ConditionalUNet, self).__init__()
        self.with_feedback = with_feedback
        
        feedback_channels = 1 if with_feedback else 0
        
        self.inc = DoubleConv(n_channels + feedback_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
        self.activation = nn.Tanh()  # Để đầu ra nằm trong khoảng [-1, 1]

    def forward(self, x, cam_feedback=None):
        # Nếu có CAM feedback, kết hợp nó với đầu vào
        if self.with_feedback and cam_feedback is not None:
            x = torch.cat([x, cam_feedback], dim=1)
        
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
        x = self.activation(x)
        
        return x

# 2. Attribute-Aware Extractor
class AttributeAwareExtractor(nn.Module):
    def __init__(self, in_channels=3, n_features=512):
        super(AttributeAwareExtractor, self).__init__()
        # Sử dụng pre-trained ResNet-18 đã loại bỏ lớp FC cuối
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Bỏ qua lớp avg pool và fc
        
        # Thêm lớp adapt nếu kênh đầu vào khác 3
        if in_channels != 3:
            self.adapt = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.adapt = nn.Identity()
        
        # Thêm lớp tích hợp đặc trưng
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(512, n_features, kernel_size=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.adapt(x)
        features = self.backbone(x)
        fused_features = self.feature_fusion(features)
        return fused_features

# 3. Predictor (Discriminator)
class Predictor(nn.Module):
    def __init__(self, in_channels=512):
        super(Predictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)

# 4. Prediction Explainer (CAM Generator)
class PredictionExplainer(nn.Module):
    def __init__(self, in_channels=512):
        super(PredictionExplainer, self).__init__()
        # Weight cho CAM
        self.cam_weights = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, features):
        # Tạo heatmap trọng số
        cam_raw = self.cam_weights(features)
        batch_size, _, h, w = cam_raw.shape
        
        # Thực hiện normalize CAM
        cam_flat = cam_raw.view(batch_size, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        
        # Normalize to [0, 1]
        cam_normalized = (cam_raw - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam_normalized

# 5. Tổng thể hệ thống
class DeepFakeRefinementSystem(nn.Module):
    def __init__(self, config, with_feedback=True):
        super(DeepFakeRefinementSystem, self).__init__()
        self.generator = ConditionalUNet(n_channels=3, n_classes=3, with_feedback=with_feedback)
        self.feature_extractor = AttributeAwareExtractor(in_channels=3, n_features=512)
        self.predictor = Predictor(in_channels=512)
        self.explainer = PredictionExplainer(in_channels=512)
        
        # VGG cho perceptual loss
        vgg = models.vgg16(pretrained=True).features[:16]  # Lấy các lớp đầu của VGG16
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        self.with_feedback = with_feedback
        self.config = config
    
    def forward(self, fake_images, real_images=None, mode='train'):
        # Bước 1: Tạo CAM feedback (nếu ở chế độ huấn luyện với feedback)
        cam_feedback = None
        if mode == 'train' and self.with_feedback:
            # Trước tiên, tạo phiên bản tinh chỉnh không có feedback
            refined_fake_without_feedback = self.generator(fake_images, None)
            
            # Trích xuất đặc trưng và dự đoán
            features_without_feedback = self.feature_extractor(refined_fake_without_feedback)
            _ = self.predictor(features_without_feedback)
            
            # Tạo CAM ban đầu
            cam_feedback = self.explainer(features_without_feedback)
        
        # Bước 2: Sinh ảnh tinh chỉnh với feedback (nếu có)
        refined_fake = self.generator(fake_images, cam_feedback)
        
        # Bước 3: Trích xuất đặc trưng
        features = self.feature_extractor(refined_fake)
        
        # Bước 4: Dự đoán Real/Fake
        predictions = self.predictor(features)
        
        # Bước 5: Giải thích dự đoán với CAM
        cam_maps = self.explainer(features)
        
        if mode == 'train':
            # Trích xuất đặc trưng từ ảnh thật cho perceptual loss
            real_features = self.feature_extractor(real_images)
            
            # Tính VGG features cho perceptual loss
            vgg_fake = self.vgg(refined_fake)
            vgg_real = self.vgg(real_images)
            
            return {
                'refined_fake': refined_fake,
                'features': features,
                'predictions': predictions,
                'cam_maps': cam_maps,
                'real_features': real_features,
                'vgg_fake': vgg_fake,
                'vgg_real': vgg_real
            }
        else:
            return {
                'refined_fake': refined_fake,
                'predictions': predictions,
                'cam_maps': cam_maps
            }

# --------------- ĐỊNH NGHĨA BỘ DỮ LIỆU ---------------

class FaceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.fake_paths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Đảm bảo có cùng số lượng ảnh
        min_len = min(len(self.real_paths), len(self.fake_paths))
        self.real_paths = self.real_paths[:min_len]
        self.fake_paths = self.fake_paths[:min_len]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.real_paths)
    
    def __getitem__(self, idx):
        real_img_path = self.real_paths[idx]
        fake_img_path = self.fake_paths[idx]
        
        # Đọc ảnh (sử dụng PIL hoặc opencv)
        from PIL import Image
        real_img = Image.open(real_img_path).convert('RGB')
        fake_img = Image.open(fake_img_path).convert('RGB')
        
        if self.transform:
            real_img = self.transform(real_img)
            fake_img = self.transform(fake_img)
        
        return {'real': real_img, 'fake': fake_img}

# --------------- HÀM LOSS ---------------

def compute_losses(outputs, config):
    # Lấy các đầu ra từ mô hình
    refined_fake = outputs['refined_fake']
    predictions = outputs['predictions']
    cam_maps = outputs['cam_maps']
    real_features = outputs['real_features']
    features = outputs['features']
    vgg_fake = outputs['vgg_fake']
    vgg_real = outputs['vgg_real']
    
    # Loss thành phần
    # 1. Adversarial loss
    adv_loss = -torch.mean(torch.log(predictions + 1e-8))
    
    # 2. Reconstruction loss (L1)
    # Ở đây chúng ta muốn refined_fake giống với ảnh thật, không phải ảnh fake
    # Nhưng trong trường hợp thực tế có thể không có ảnh thật tương ứng, bạn có thể dùng ảnh gốc
    rec_loss = F.l1_loss(refined_fake, real_fake_pairs['real'])
    
    # 3. Perceptual loss
    percep_loss = F.mse_loss(vgg_fake, vgg_real)
    
    # 4. CAM feedback loss (tập trung vào vùng nghi ngờ)
    # Normalize CAM để trọng số nằm trong [0, 1]
    pixel_diff = torch.abs(refined_fake - real_fake_pairs['real'])
    cam_feedback_loss = torch.mean(cam_maps * pixel_diff)
    
    # Feature matching loss (thêm vào)
    feature_matching_loss = F.mse_loss(features, real_features)
    
    # Tính tổng loss với các trọng số
    total_loss = (config.lambda_adv * adv_loss + 
                  config.lambda_rec * rec_loss + 
                  config.lambda_percep * percep_loss + 
                  config.lambda_cam * cam_feedback_loss + 
                  0.5 * feature_matching_loss)
    
    return {
        'total_loss': total_loss,
        'adv_loss': adv_loss,
        'rec_loss': rec_loss,
        'percep_loss': percep_loss,
        'cam_feedback_loss': cam_feedback_loss,
        'feature_matching_loss': feature_matching_loss
    }

def compute_discriminator_loss(real_features, fake_features, real_preds, fake_preds):
    # Binary Cross Entropy Loss cho Discriminator
    real_loss = -torch.mean(torch.log(real_preds + 1e-8))
    fake_loss = -torch.mean(torch.log(1 - fake_preds + 1e-8))
    
    # Feature matching loss
    feature_matching_loss = F.mse_loss(fake_features, real_features.detach())
    
    total_loss = real_loss + fake_loss + 0.1 * feature_matching_loss
    
    return {
        'total_loss': total_loss,
        'real_loss': real_loss,
        'fake_loss': fake_loss,
        'feature_matching_loss': feature_matching_loss
    }

# --------------- CHỨC NĂNG TRAINING ---------------

def train_model(model, dataloader, config):
    # Khởi tạo optimizers
    generator_params = list(model.generator.parameters()) 
    discriminator_params = list(model.feature_extractor.parameters()) + \
                          list(model.predictor.parameters()) + \
                          list(model.explainer.parameters())
    
    g_optimizer = optim.Adam(generator_params, lr=config.lr, betas=(config.beta1, config.beta2))
    d_optimizer = optim.Adam(discriminator_params, lr=config.lr, betas=(config.beta1, config.beta2))
    
    g_scheduler = StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = StepLR(d_optimizer, step_size=30, gamma=0.5)
    
    global_step = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch_idx, real_fake_pairs in enumerate(progress_bar):
            real_images = real_fake_pairs['real'].to(config.device)
            fake_images = real_fake_pairs['fake'].to(config.device)
            
            batch_size = real_images.size(0)
            
            # --------------- Huấn luyện Discriminator ---------------
            d_optimizer.zero_grad()
            
            # Trích xuất đặc trưng từ ảnh thật
            real_features = model.feature_extractor(real_images)
            real_preds = model.predictor(real_features)
            
            # Tạo và phân tích ảnh fake tinh chỉnh (với detach để không cập nhật Generator)
            with torch.no_grad():
                refined_fake = model.generator(fake_images, None)
            
            # Trích xuất đặc trưng và dự đoán từ ảnh fake tinh chỉnh
            fake_features = model.feature_extractor(refined_fake.detach())
            fake_preds = model.predictor(fake_features)
            
            # Tính loss cho discriminator
            d_losses = compute_discriminator_loss(real_features, fake_features, real_preds, fake_preds)
            d_loss = d_losses['total_loss']
            
            d_loss.backward()
            d_optimizer.step()
            
            # --------------- Huấn luyện Generator với CAM feedback ---------------
            g_optimizer.zero_grad()
            
            # Forward qua mô hình hoàn chỉnh với CAM feedback
            outputs = model(fake_images, real_images, mode='train')
            
            # Tính loss
            g_losses = compute_losses(outputs, config)
            g_loss = g_losses['total_loss']
            
            g_loss.backward()
            g_optimizer.step()
            
            # Cập nhật loss và thanh tiến trình
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            progress_bar.set_postfix({
                'G_loss': g_loss.item(), 
                'D_loss': d_loss.item(),
                'Adv': g_losses['adv_loss'].item(),
                'Rec': g_losses['rec_loss'].item(),
                'CAM': g_losses['cam_feedback_loss'].item()
            })
            
            # Lưu hình ảnh mẫu
            if global_step % config.sample_interval == 0:
                save_sample_images(real_images, fake_images, outputs, global_step, config)
            
            global_step += 1
            
        # Kết thúc mỗi epoch
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{config.epochs}, Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
        
        # Cập nhật learning rate scheduler
        g_scheduler.step()
        d_scheduler.step()
        
        # Lưu checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(model, g_optimizer, d_optimizer, epoch, global_step, config)

def save_sample_images(real_images, fake_images, outputs, step, config):
    # Tạo grid hình ảnh mẫu
    refined_fake_images = outputs['refined_fake']
    cam_maps = outputs['cam_maps']
    
    # Chọn 4 hình ảnh mẫu đầu tiên để hiển thị
    n_samples = min(4, real_images.size(0))
    
    # Tạo grid để so sánh
    comparison = torch.cat([
        real_images[:n_samples],
        fake_images[:n_samples],
        refined_fake_images[:n_samples],
        cam_maps[:n_samples].repeat(1, 3, 1, 1)  # Chuyển đổi CAM thành 3 kênh để hiển thị
    ], dim=0)
    
    save_image(comparison.detach(), 
               os.path.join(config.result_dir, f'comparison_{step}.png'),
               nrow=n_samples, normalize=True)
    
    # Lưu heatmap của CAM riêng biệt
    plt.figure(figsize=(12, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        cam_np = cam_maps[i, 0].detach().cpu().numpy()
        plt.imshow(cam_np, cmap='jet')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.result_dir, f'cam_heatmap_{step}.png'))
    plt.close()

def save_checkpoint(model, g_optimizer, d_optimizer, epoch, step, config):
    checkpoint = {
        'generator_state_dict': model.generator.state_dict(),
        'extractor_state_dict': model.feature_extractor.state_dict(),
        'predictor_state_dict': model.predictor.state_dict(),
        'explainer_state_dict': model.explainer.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def load_checkpoint(model, g_optimizer, d_optimizer, checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.feature_extractor.load_state_dict(checkpoint['extractor_state_dict'])
    model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
    model.explainer.load_state_dict(checkpoint['explainer_state_dict'])
    
    g_optimizer.load_state_dict(checkpoint['g_optimizer'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['step']
    
    return start_epoch, global_step

# --------------- HÀM MAIN ---------------

def main():
    parser = argparse.ArgumentParser(description="DeepFake Refinement with CAM Feedback")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real face images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory containing fake face images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--with_feedback", action="store_true", help="Use CAM feedback in training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Khởi tạo cấu hình
    config = Config()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    
    create_dirs(config)
    
    # Định nghĩa transformations
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Tạo dataset và dataloader
    dataset = FaceDataset(args.real_dir, args.fake_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    # Khởi tạo mô hình
    model = DeepFakeRefinementSystem(config, with_feedback=args.with_feedback).to(config.device)
    
    # Khởi tạo optimizers
    generator_params = list(model.generator.parameters()) 
    discriminator_params = list(model.feature_extractor.parameters()) + \
                          list(model.predictor.parameters()) + \
                          list(model.explainer.parameters())
    
    g_optimizer = optim.Adam(generator_params, lr=config.lr, betas=(config.beta1, config.beta2))
    d_optimizer = optim.Adam(discriminator_params, lr=config.lr, betas=(config.beta1, config.beta2))
    
    # Load checkpoint nếu có
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, g_optimizer, d_optimizer, args.resume, config)
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")
    
    # Huấn luyện mô hình
    train_model(model, dataloader, config)

if __name__ == "__main__":
    main()