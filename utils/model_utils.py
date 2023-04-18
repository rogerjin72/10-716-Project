import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
            padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, 
            padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=5)
        self.dropout3 = nn.Dropout(0.2)

        flatten = torch.nn.Flatten(1, -1)
        self.linear = nn.Linear(128, 10)

        self.model = nn.Sequential(
            self.conv1, 
            self.relu1, 
            self.pool1, 
            self.dropout1,
            self.conv2, 
            self.relu2, 
            self.pool2, 
            self.dropout2,
            self.conv3, 
            self.relu3,
            self.pool3,
            self.dropout3,
            flatten,
            self.linear
        )
    
    def forward(self, x):
        return self.model(x)


    def get_intermediate(self, x):
        intermediate = nn.Sequential(
            self.conv1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.relu2,
            self.pool2
        )
        return intermediate(x)


class ConceptModel(nn.Module):
    def __init__(self, n_concepts, beta=0.1):
        super().__init__()
        self.n_concepts = n_concepts
        self.indim = 64

        self.concepts = nn.Linear(self.indim, n_concepts)
        self.fc = nn.Linear(49 * n_concepts, 128)
        self.relu = nn.ReLU()
        self.beta = beta

    def forward(self, x):
        x = self.concepts(x)
        x = F.threshold(x, self.beta, 0)
        x = F.normalize(x, dim=-1)
        x = x.flatten(1)
        x = self.fc(x)
        return self.relu(x)
    
    def get_concepts(self):
        return self.concepts(torch.eye(64)).T
