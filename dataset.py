import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config

# Color augmentations applied to training tiles only.
# Grayscale images are replicated to 3 channels, so saturation/hue are skipped.
TRAIN_AUGMENTS = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4),
    T.RandomAutocontrast(p=0.3),
    T.RandomEqualize(p=0.2),
])


class TileDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        with open(config.SPLITS_FILE) as f:
            index = json.load(f)
        self.tiles = index[split]
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        img = Image.open(tile['tile_path']).convert('L')
        img = Image.merge('RGB', (img, img, img))

        if self.augment:
            img = TRAIN_AUGMENTS(img)

        img_t = TF.to_tensor(img)  # [3, H, W] float32 in [0, 1]

        boxes  = torch.tensor(tile['boxes'],  dtype=torch.float32)
        labels = torch.tensor(tile['labels'], dtype=torch.int64)

        if boxes.numel() == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([idx]),
        }

        return img_t, target
