import os
import torch
from tqdm import tqdm
from itertools import cycle
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
from models.unet_generator import UNetGenerator
from models.discriminator import FaceDiscriminator
from models.attribute_head import AttributeHeader
from models.cam import compute_cam
from dataset import FaceDataset, split_dataset

@dataclass
class Args:
    epochs: int = 5
    batch_size: int = 2
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 164
    real_folder: str = "datasets/images/FakeAVCeleb/real"
    fake_folder: str = "datasets/images/FakeAVCeleb/fake"
    att_bboxs_path: str = "datasets/att_bboxs.json"
    save_dir: str = "checkpoints"

class AdversarialTrainer:
    def __init__(self, args: Args):
        self.args = args
        self.G = UNetGenerator().to(args.device)
        self.D = FaceDiscriminator().to(args.device)
        self.A = AttributeHeader().to(args.device)
        
        self.opt_G = torch.optim.AdamW(self.G.parameters(), lr=args.lr_g, weight_decay=1e-4)
        self.opt_D = torch.optim.AdamW(self.D.parameters(), lr=args.lr_d, weight_decay=1e-4)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        # Load datasets
        self.real_dataset = FaceDataset(args.real_folder, args.image_size)
        self.fake_dataset = FaceDataset(args.fake_folder, args.image_size)
        real_train, real_val = split_dataset(self.real_dataset, val_ratio=0.2)
        fake_train, fake_val = split_dataset(self.fake_dataset, val_ratio=0.2)

        self.real_train_loader = DataLoader(real_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.fake_train_loader = DataLoader(fake_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.real_val_loader   = DataLoader(real_val, batch_size=args.batch_size, shuffle=False)
        self.fake_val_loader   = DataLoader(fake_val, batch_size=args.batch_size, shuffle=False)

    def _label(self, size, value):
        label = torch.zeros((size,2), dtype=torch.float32, device=self.args.device)
        label[:, value] = 1.0
        return label

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            real_cycle = cycle(self.real_train_loader)
            for i, (fake_img, fake_att_gt) in tqdm(enumerate(self.fake_train_loader)):
                real_img, real_att_gt = next(real_cycle)
                real_img = real_img.to(self.args.device)
                fake_img = fake_img.to(self.args.device)    
                real_att_gt = real_att_gt.to(self.args.device)
                fake_att_gt = fake_att_gt.to(self.args.device)

                # ---------------------- Generator forward ----------------------
                # No CAM available yet ⇒ zeros tensor acts as neutral map
                zero_cam = torch.zeros(fake_img.size(0), 1, fake_img.size(2), fake_img.size(3), device=fake_img.device)
                refined = self.G(fake_img, zero_cam)

                # ---------------- Discriminator forward / update ---------------
                self.opt_D.zero_grad(set_to_none=True)
                logits_real, feat_real = self.D(real_img)
                logits_fake, feat_fake = self.D(refined.detach())
                
                logits_attributes_real = self.A(feat_real)
                logits_attributes_fake = self.A(feat_fake)
                
                loss_D = self.bce(logits_real, self._label(real_img.size(0), 0)) + \
                         self.bce(logits_fake, self._label(fake_img.size(0), 1)) + \
                         F.smooth_l1_loss(logits_attributes_real, real_att_gt) + \
                         F.smooth_l1_loss(logits_attributes_fake, fake_att_gt)
                        
                loss_D.backward()
                self.opt_D.step()

                # ------------- Compute CAM on *current* refined batch ----------
                cam = compute_cam(self.D, refined, class_idx=1)
                cam = F.interpolate(cam, size=refined.shape[2:], mode="bilinear", align_corners=False)

                # ------------------ Generator update (with CAM) ----------------
                self.opt_G.zero_grad(set_to_none=True)
                refined2 = self.G(fake_img, cam)  # second pass with feedback
                logits_fake2, _ = self.D(refined2)

                loss_G = self.bce(logits_fake2, self._label(fake_img.size(0), 0)) + self.mse(refined2, fake_img)
                loss_G.backward()
                self.opt_G.step()

                if i % 25 == 0:
                    print(f"[Epoch {epoch}/{self.args.epochs}] Step {i:03d}| "
                          f"loss_D = {loss_D.item():.4f} | loss_G = {loss_G.item():.4f}")
            
            self.validate_accuracy()
            # Save checkpoints at epoch end
            torch.save({"G": self.G.state_dict(), "D": self.D.state_dict()},
                       os.path.join(self.args.save_dir, f"epoch_{epoch}.pth"))
            print("Saved checkpoint for epoch", epoch)
            
    @torch.no_grad()
    def validate_accuracy(self):
        self.D.eval()
        correct, total = 0, 0
        for real in self.real_val_loader:
            real = real.to(self.args.device)
            logits, _ = self.D(real)
            pred = logits.argmax(dim=1)
            correct += (pred == 0).sum().item()
            total += real.size(0)
        for fake in self.fake_val_loader:
            fake = fake.to(self.args.device)
            logits, _ = self.D(fake)
            pred = logits.argmax(dim=1)
            correct += (pred == 1).sum().item()
            total += fake.size(0)
        acc = correct / total if total > 0 else 0
        print(f"Validation Discriminator Accuracy: {acc:.4f}")
        self.D.train()
        return acc
            
if __name__ == "__main__":
    args = Args()
    trainer = AdversarialTrainer(args)
    trainer.train()