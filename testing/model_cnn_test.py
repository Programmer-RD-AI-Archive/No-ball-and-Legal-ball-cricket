import torch.nn.functional as F
import torch
import torch.nn as nn
from config import *


class CNN_Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.conv1 = nn.Conv2d(config["testing"]["color_channel(s)"], 32, 5)
        self.conv1batchnorn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2batchnorn = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv3batchnorn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(config["testing"]["convtofc"], 256)
        self.fc1batchnorn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2batchnorn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, config["testing"]["NUM_CLASS"])

    def forward(self, X, shape=False):
        preds = self.conv1(X)
        preds = self.conv1batchnorn(preds)
        preds = self.activation(preds)
        preds = self.conv2(preds)
        preds = self.activation(preds)
        preds = self.conv3(preds)
        preds = self.conv3batchnorn(preds)
        preds = self.activation(preds)
        if shape:
            print(preds.shape)
        config["testing"]["convtofc"] = int(
            preds.shape[1] * preds.shape[2] * preds.shape[3]
        )
        preds = preds.view(-1, config["testing"]["convtofc"])
        preds = self.fc1(preds)
        preds = self.fc1batchnorn(preds)
        preds = self.activation(preds)
        preds = self.fc2(preds)
        preds = self.fc2batchnorn(preds)
        preds = self.activation(preds)
        preds = self.fc3(preds)
        return preds
