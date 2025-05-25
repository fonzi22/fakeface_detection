import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import os
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# =========================
# Dataset
# =========================

class ContrastiveDeepfakeDataset(Dataset):
    def __init__(self, fdata_folder, transform=None):
        """
        face_img_paths: list of file paths to face images
        attr_img_paths: list of file paths to attribute images (same order as face_img_paths)
        labels: list of 0 (real) or 1 (fake), same order
        transform: torchvision transforms to apply
        """
        self.folder = data_folder
        self.transform = transform
        self.real_folder = os.path.join(fdata_folder, 'real_attribute/face')
        self.fake_folder = os.path.join(fdata_folder, 'fake_attribute/face')
        self.face_img_paths = []
        self.attr_img_paths = []
        self.labels = []
        self.label_to_indices = {}

        i = 0
        # Build label to indices mapping for negative sampling
        for dirpath, dirnames, filenames in os.walk(self.real_folder):
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    face_path = os.path.join(dirpath, filename)
                    self.face_img_paths.append(face_path)
                    self.attr_img_paths.append(str.replace(face_path, 'real_attribute/face', 'real_attribute/masked_face'))
                    self.labels.append(0)
                    i += 1
                    if i > 1000:
                        break            
        
        for dirpath, dirnames, filenames in os.walk(self.fake_folder):
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    face_path = os.path.join(dirpath, filename)
                    self.face_img_paths.append(face_path)
                    self.attr_img_paths.append(str.replace(face_path, 'fake_attribute/face', 'fake_attribute/masked_face'))
                    self.labels.append(1)
                    i += 1
                    if i > 1000:
                        break            
                    
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.face_img_paths)

    def __getitem__(self, idx):
        # Anchor: face image
        face_img = Image.open(self.face_img_paths[idx]).convert('RGB')
        # Positive: attribute image (same sample)
        attr_img = Image.open(self.attr_img_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # Negative: face image with different label
        neg_label = 1 - label
        neg_idx = random.choice(self.label_to_indices[neg_label])
        neg_face_img = Image.open(self.face_img_paths[neg_idx]).convert('RGB')

        if self.transform:
            face_img = self.transform(face_img)
            attr_img = self.transform(attr_img)
            neg_face_img = self.transform(neg_face_img)

        return {
            'face_img': face_img,         # Anchor
            'attr_img': attr_img,         # Positive
            'neg_face_img': neg_face_img, # Negative
            'label': torch.tensor(label, dtype=torch.long)
        }
        
# =========================
class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AttrEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =========================
# Contrastive Loss
# =========================

def contrastive_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

# =========================
# Training Loop
# =========================

def train_contrastive(
    face_encoder, attr_encoder, dataloader, optimizer, device, epochs=10, margin=1.0
):
    face_encoder.train()
    attr_encoder.train()
    # Add tqdm progress bar to training loop
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                face_img = batch['face_img'].to(device)
                attr_img = batch['attr_img'].to(device)
                neg_face_img = batch['neg_face_img'].to(device)

                optimizer.zero_grad()
                anchor_emb = face_encoder(face_img)
                pos_emb = attr_encoder(attr_img)
                neg_emb = face_encoder(neg_face_img)

                loss = contrastive_loss(anchor_emb, pos_emb, neg_emb, margin)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# =========================
# Example Usage
# =========================

if __name__ == "__main__":
    # Example file lists (replace with your own)
    data_folder = "./datasets/images/FakeAVCeleb"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ContrastiveDeepfakeDataset(data_folder, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_encoder = FaceEncoder().to(device)
    attr_encoder = AttrEncoder().to(device)

    optimizer = torch.optim.Adam(list(face_encoder.parameters()) + list(attr_encoder.parameters()), lr=1e-4)

    train_contrastive(face_encoder, attr_encoder, dataloader, optimizer, device, epochs=10)
    
 