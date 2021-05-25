import torch.nn.functional as F
import torch
import torch.nn as nn
from config import *


class Model_Clf_and_Conv1d_Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 5)
        self.conv1batchnorn = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.conv2batchnorn = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 5)
        self.conv3batchnorn = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128*12532, 256)
        self.fc1batchnorn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2batchnorn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, config["testing"]["NUM_CLASS"])
        self.activation = F.relu

    def forward(self, X, shape=False):
        X = X.view(
            -1,
            3,
            config["testing"]["img_size"] * config["testing"]["img_size"],
        )
        preds = self.conv1(X)
        preds = self.conv1batchnorn(preds)
        preds = self.activation(preds)
        preds = self.conv2(preds)
        preds = self.conv2batchnorn(preds)
        preds = self.activation(preds)
        preds = self.conv3(preds)
        preds = self.conv3batchnorn(preds)
        preds = self.activation(preds)
        if shape:
            print(preds.shape)
        preds = preds.view(-1,128*12532)
        preds = self.fc1(preds)
        preds = self.fc1batchnorn(preds)
        preds = self.activation(preds)
        preds = self.fc2(preds)
        preds = self.fc2batchnorn(preds)
        preds = self.activation(preds)
        preds = self.fc3(preds)
        return preds
