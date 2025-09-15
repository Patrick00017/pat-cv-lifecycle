from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import pytorch_lightning as pl
import torchvision
import os

data_root_path = "datasets/"

# def get_mnist_train_loader(batch_size: int):
#     transform = transforms.ToTensor()
#     dataset = MNIST(data_root_path, download=True, transform=transform)
#     train_loader = DataLoader(dataset, batch_size=batch_size)
#     return train_loader

class MNISTDataModule(pl.LightningModule):
    def __init__(self, dataset_dir, batch_size):
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
    
    

def get_cifar10_datamodule(batch_size: int):
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    cifar10_dm = CIFAR10DataModule(
        data_dir=data_root_path,
        batch_size=batch_size,
        num_workers=2,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    return cifar10_dm
    