import torch
import torchvision

from utils.config import *
from torch.utils.data import DataLoader

def get_mnist(fp, train=True, batch_size=64):
    mnist_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )    

    dataset = torchvision.datasets.MNIST(fp, train, download=True, transform=mnist_transforms)
    return DataLoader(dataset, batch_size, shuffle=not train)