import os
import cv2
import torch

import config as CFG
import torchvision.transforms.v2 as transforms
from PIL import Image


class DeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transforms):
        """
        image_paths and cpations must have the same length; so, if there are
        multiple captions for each image, the image_paths must have repetitive
        file names 
        """

        self.image_paths = image_paths
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}

        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transforms(image)
        item['image'] = image
        item['label'] = torch.tensor([1, 0]) if self.label[idx] else torch.tensor([0, 1])

        return item


    def __len__(self):
        return len(self.image_paths)

def get_transforms(mode="train"):
    return transforms.Compose(
        [
            transforms.Resize((CFG.size, CFG.size)),  # Always applied
            transforms.ToTensor(),                    # Always applied
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Always applied
        ]
    )

    