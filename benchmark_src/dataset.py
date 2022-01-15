import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import rasterio
import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor
import numpy as np


def pre_transforms(image_size=512):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        # albu.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=BORDER_CONSTANT),
        albu.Resize(height=512, width=512, always_apply=True),
        albu.CLAHE(p=1, always_apply=True),
        # albu.RandomCrop(height=image_size, width=image_size, always_apply=True),
    ]

    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    # albu.Normalize(),
    return [albu.Resize(512, 512, always_apply=True), albu.Normalize(), ToTensor()]

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_valid_transforms():
    return compose([
        pre_transforms(),
        post_transforms()
    ])

class CloudDataset(Dataset):
    def __init__(self, items, augmentations=get_valid_transforms):
        self.items = items
        self.augmentations = augmentations

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        b02_file = item["b02"]
        b03_file = item["b03"]
        b04_file = item["b04"]
        
        f = rasterio.open(b02_file)
        b02 = (np.sqrt(f.read(1))).astype(np.uint8)

        f = rasterio.open(b03_file)
        b03 = (np.sqrt(f.read(1))).astype(np.uint8)

        f = rasterio.open(b04_file)
        b04 = (np.sqrt(f.read(1))).astype(np.uint8)

        img = np.dstack([b02, b03, b04])

        if self.augmentations:
            sample = self.augmentations(image=img)
            img = sample["image"]
        
        
        return img.float()