from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

def get_mnist_train_loader():
    dataset = MNIST(os.getcwd(), download=True)
    transform = transforms.ToTensor()
    train_loader = DataLoader(dataset)
    return train_loader
    