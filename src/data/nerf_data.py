from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
import os

data_root_path = "datasets/"

class MNISTDataModule(pl.LightningModule):
    def __init__(self, batch_size, dataset_dir=data_root_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
    
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