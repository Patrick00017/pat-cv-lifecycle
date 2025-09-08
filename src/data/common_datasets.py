from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

data_root_path = "datasets/"

def get_mnist_train_loader():
    transform = transforms.ToTensor()
    dataset = MNIST(data_root_path, download=True, transform=transform)
    train_loader = DataLoader(dataset)
    return train_loader
    