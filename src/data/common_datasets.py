from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision
import os

data_root_path = "datasets/"

def get_mnist_train_loader(batch_size: int):
    transform = transforms.ToTensor()
    dataset = MNIST(data_root_path, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    return train_loader

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
    