from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
import torch
import json

def split_dataset(dataset, val_ratio=0.2, seed=42):
    total = len(dataset)
    idxs = list(range(total))
    train_idx, val_idx = train_test_split(idxs, test_size=val_ratio, random_state=seed, shuffle=True)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

class FaceDataset(Dataset):
    # def __init__(self, img_dir, att_bboxs_path, label=0, image_size=256, transform=None):
    def __init__(self, img_dir,  image_size=256, transform=None):
        self.img_paths = [f for f in Path(img_dir).rglob("*") if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        # with open(att_bboxs_path, 'r') as f:
        #     self.att_bboxs = json.load(f)
        self.size = image_size
        # self.label = label
        self.transform = transform if transform is not None else self.get_transforms()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # bbox = []
        # if str(img_path) in self.att_bboxs.keys():
        #     for att_name, coords in self.att_bboxs[str(img_path)].items():
        #         x1, y1, x2, y2 = coords
        #         x1 /= img.width
        #         y1 /= img.height
        #         x2 /= img.width
        #         y2 /= img.height
        #         bbox.append([x1, y1, x2, y2])
        
        if self.transform:
            img = self.transform(img)
            
        # if self.label == 0:
        #     label = torch.tensor([1, 0], dtype=torch.long)
        # else:
        #     label = torch.tensor([0, 1], dtype=torch.long)
            
        # return {'image': img, 'label': label, 'bbox': torch.tensor(bbox, dtype=torch.float32) if bbox else torch.empty((0, 4))}
        return img
        
    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
