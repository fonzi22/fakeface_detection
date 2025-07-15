import torch
import os
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def split_dataset(dataset: Dataset, val_ratio: float = 0.2, seed: int = 42):
    """Split a Dataset into train/val Subsets."""
    idxs = list(range(len(dataset)))
    train_idxs, val_idxs = train_test_split(
        idxs, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)


class FaceDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        image_size: tuple[int,int] = (256, 256),
        crop_size: tuple[int,int] = (224, 224),
    ):
        self.img_paths = []
        self.anno_paths = []
        
        for img_path in Path(img_dir).rglob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                anno_path = str(img_path).replace('.jpg', '.npy').replace('.png', '.npy').replace('images', 'annotations')
                
                if os.path.exists(anno_path):
                    self.img_paths.append(img_path)
                    self.anno_paths.append(anno_path)
       
        self.transform = self._get_transforms(image_size, crop_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        anno_path = self.anno_paths[idx]
        # Load image & normalized boxes ---
        img = np.array(Image.open(img_path).convert("RGB"))

        bbox_norm = np.load(str(anno_path))  # e.g. [[xmin,ymin,xmax,ymax], ...]
        h0, w0 = img.shape[:2]

        # Convert normalized VOC → absolute pixels ---
        bboxes_abs = [
            [xmin * w0, ymin * h0, xmax * w0, ymax * h0]
            for xmin, ymin, xmax, ymax in bbox_norm
        ]
        
        labels = [1] * len(bboxes_abs)  # Dummy labels, assuming all boxes are valid

        # Apply Albumentations pipeline ---
        augmented = self.transform(
            image=img, bboxes=bboxes_abs, labels=labels
        )
        img_aug = augmented["image"]
        boxes_abs_aug = augmented["bboxes"]

        # Convert back to normalized VOC over new image size ---
        _, h2, w2 = img_aug.shape  # (C, H, W)
        boxes_norm_aug = [
            [x1 / w2, y1 / h2, x2 / w2, y2 / h2]
            for x1, y1, x2, y2 in boxes_abs_aug
        ]

        return img_aug, torch.tensor(boxes_norm_aug, dtype=torch.float32)

    def _get_transforms(self, image_size, crop_size):
        return A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.1, p=0.8
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.5
                ),
                A.ImageCompression(quality_range=(70, 100), p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.Blur(blur_limit=(3, 7), p=0.5),
                ], p=0.5),
                A.Downscale(p=0.25),
                A.Normalize(),         
                ToTensorV2(),           
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",  
                label_fields=["labels"],
                min_visibility=0.0     
            ),
        )
