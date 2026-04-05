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

        # Membaca 10 Band Sentinel-2 menggunakan rasterio
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) # Format: (Channels, Height, Width)
        
        # Membaca Mask (biasanya 1 channel)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32) # Format: (Height, Width)

        # Konversi ke Torch Tensor
        # Karena nilai sudah 0-1, kita tidak perlu pembagian lagi
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long() # Gunakan .long() jika untuk klasifikasi (segmentasi)

        return image, mask

class SentinelDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, mask_dir, batch_size=16, train_split=0.8):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.train_split = train_split

    def setup(self, stage=None):
        full_dataset = SentinelDataset(self.image_dir, self.mask_dir)
        total_count = len(full_dataset)

        # Menentukan proporsi (Misal: 80% Train, 10% Val, 10% Test)
        train_size = int(0.8 * total_count)
        val_size = int(0.1 * total_count)
        test_size = total_count - train_size - val_size # Sisanya untuk test

        # Split data secara acak sekali saja
        # Gunakan generator dengan seed agar pembagiannya konsisten setiap kali running
        seed = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset, [train_size, val_size, test_size], generator=seed
        )

        # Logika 'stage' digunakan oleh Trainer untuk memanggil yang diperlukan saja
        if stage == "fit" or stage is None:
            # Train & Val digunakan saat trainer.fit()
            pass 
        
        if stage == "test":
            # Test digunakan saat trainer.test()
            pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)