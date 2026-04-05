import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import rasterio
import numpy as np
import os
from pathlib import Path

class SentinelDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Reading Sentinel-2 Bands using rasterio
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) # Format: (Channels, Height, Width)
        
        # Reading Mask data
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32) # Format: (Height, Width)

        # Convert to Torch Tensor
        # Since the value is already 0-1, we don't need to divide anymore.
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long() 

        return image, mask

class SentinelDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, train_mask_dir, test_dir, test_mask_dir, batch_size=16, val_split=0.2):
        super().__init__()
        self.train_dir = train_dir
        self.train_mask_dir = train_mask_dir
        self.test_dir = test_dir
        self.test_mask_dir = test_mask_dir
        self.batch_size = batch_size
        self.val_split = val_split  

    def setup(self, stage=None):
        # For the training and validation stage
        if stage == 'fit' or stage is None:
            full_train_ds = SentinelDataset(self.train_dir, self.train_mask_dir)
            
            val_size = int(self.val_split * len(full_train_ds))
            train_size = len(full_train_ds) - val_size
            
            self.train_ds, self.val_ds = random_split(
                full_train_ds, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )

        # For the testing stage
        if stage == 'test' or stage is None:
            self.test_ds = SentinelDataset(self.test_dir, self.test_mask_dir)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)