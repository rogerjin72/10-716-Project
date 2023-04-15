import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
            padding=2)
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, 
            padding=2)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        relu3 = nn.ReLU()
        pool3 = nn.MaxPool2d(kernel_size=5)
        dropout3 = nn.Dropout(0.2)

        flatten = torch.nn.Flatten(1, -1)
        self.linear = nn.Linear(128, 10)

        self.model = nn.Sequential(
            self.conv1, 
            relu1, 
            pool1, 
            dropout1,
            self.conv2, 
            relu2, 
            pool2, 
            dropout2,
            self.conv3, 
            relu3,
            pool3,
            dropout3,
            flatten,
            self.linear
        )
    
    def forward(self, x):
        return self.model(x)

    