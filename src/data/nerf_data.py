from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
import torch
from typing import Optional, Tuple, List, Union, Callable
from einops import rearrange, repeat

data_root_path = "datasets/"

class NeRFDataset(Dataset):
    def __init__(self, dataset_dir, n_training=100, testimg_idx=101):
        self.dataset_dir = dataset_dir
        self.n_training = n_training

        data = np.load(f"{self.dataset_dir}/tiny_nerf_data.npz")
        self.images = torch.from_numpy(data['images'])
        self.poses = torch.from_numpy(data['poses'])
        self.focal = torch.from_numpy(data['focal'])
        self.testimg = torch.from_numpy(data["images"][testimg_idx])
        self.testpose = torch.from_numpy(data["poses"][testimg_idx])
        # image info
        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2., 6.

    def get_rays(self, height, width, focal_length, c2w):
        r"""
        get rays original and viewdirs
        output shape: rays_o (100, 100, 3) rays_d (100, 100, 3)
        """
        i, j = torch.meshgrid(
            torch.arange(100, dtype=torch.float32),
            torch.arange(100, dtype=torch.float32),
            indexing='ij'
        )
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        focal_length = 10
        # trick: create pinhole 
        directions = torch.stack([(i - width * 0.5) / focal_length, -(j - height * 0.5) / focal_length, -torch.ones_like(i)], dim=-1)
        print(directions.shape)
        # convert local direction to global direction
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def __len__(self):
        return self.n_training

    def __getitem__(self, idx):
        target_img_idx = np.random.randint(self.images.shape[0])
        target_img = self.images[target_img_idx]
        height, width = target_img.shape[:2]
        target_pose = self.poses[target_img_idx]
        rays_o, rays_d = self.get_rays(height, width, self.focal, target_pose)
        rays_o = rearrange(rays_o, "h w c -> (h w) c")
        rays_d = rearrange(rays_d, "h w c -> (h w) c")
        target_img = rearrange(target_img, "h w c -> (h w) c")
        return rays_o, rays_d, target_img

class NeRFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, n_training=100, testimg_idx=101, dataset_dir=data_root_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.n_training = n_training

    def setup(self, stage=None):
        # print(self.img_size)
        if stage == 'fit' or stage is None:
            self.train_dataset = NeRFDataset(self.dataset_dir)
            self.val_dataset = NeRFDataset(self.dataset_dir)
        if stage == 'test' or stage is None:
            self.test_dataset = NeRFDataset(self.dataset_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)