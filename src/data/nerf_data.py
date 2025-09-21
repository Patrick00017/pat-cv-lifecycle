from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
import os
import numpy as np
import torch
from typing import Optional, Tuple, List, Union, Callable

data_root_path = "datasets/"

class NeRFModule(pl.LightningModule):
    def __init__(self, batch_size, n_training=100, testimg_idx=101, dataset_dir=data_root_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.n_training = n_training

        # load nerf data
        data = np.load(f"{self.dataset_dir}/tiny_nerf_data.npz")
        self.images = torch.from_numpy(data['images'])
        self.poses = torch.from_numpy(data['poses'])
        self.focal = torch.from_numpy(data['focal'])
        self.testimg = torch.from_numpy(data["images"][testimg_idx])
        self.testpose = torch.from_numpy(data["poses"][testimg_idx])
        # image info
        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2., 6.
    
    def get_chunks(
        self,
        inputs: torch.Tensor,
        chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Divide an input into chunks.
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    def prepare_chunks(
        self,
        points: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        points = points.reshape((-1, 3))
        points = encoding_function(points)
        points = self.get_chunks(points, chunksize=chunksize)
        return points

    def prepare_viewdirs_chunks(
        self,
        points: torch.Tensor,
        rays_d: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify viewdirs to prepare for NeRF model.
        """
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        viewdirs = encoding_function(viewdirs)
        viewdirs = self.get_chunks(viewdirs, chunksize=chunksize)
        return viewdirs
    
    def setup(self, stage=None):
        # print(self.img_size)
        if stage == 'fit' or stage is None:
            self.train_dataset = MNIST(self.dataset_dir, download=True, transform=self.transform, train=True)
            self.val_dataset = MNIST(self.dataset_dir, download=True, transform=self.transform, train=False)
        if stage == 'test' or stage is None:
            self.test_dataset = MNIST(self.dataset_dir, download=True, transform=self.transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)