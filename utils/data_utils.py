import torch
import torchvision

from utils.config import *
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_mnist(fp, train=True, shuffle=True, batch_size=64):
    mnist_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )    
    
    dataset = torchvision.datasets.MNIST(fp, train, download=True, transform=mnist_transforms)

    return DataLoader(dataset, batch_size, shuffle=shuffle and train)


def get_intermediate(fp, model, train=True):
    try:
        intermediate = torch.load(DATA_PATH / 'intermediate.pt')
    except FileNotFoundError:
        dataloader = get_mnist(fp, train, False)
        intermediate = []
    
        model.eval()
        with torch.no_grad():
            for x, _ in tqdm(dataloader):
                intermediate.append(model.get_intermediate(x))
        intermediate = torch.concat(intermediate)
        torch.save(intermediate, DATA_PATH / 'intermediate.pt')
    
    return intermediate