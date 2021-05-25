import torch.nn.functional as F
import torch
import torch.nn as nn
from config import *


class Model_Clf_Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.fc1 = nn.Linear(
            config["testing"]["color_channel(s)"]
            * config["testing"]["img_size"]
            * config["testing"]["img_size"],
            64,
        )
        self.fc1batchnorm = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.fc2batchnorm = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.fc3batchnorm = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc4batchnorm = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, config["testing"]["NUM_CLASS"])
        self.fc5batchnorm = nn.BatchNorm1d(128)

    def forward(self, X):
        X = X.view(
            -1,
            config["testing"]["color_channel(s)"]
            * config["testing"]["img_size"]
            * config["testing"]["img_size"],
        )
        preds = self.fc1(X)
        preds = self.fc1batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc2(preds)
        preds = self.fc2batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc3(preds)
        preds = self.fc3batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc4(preds)
        preds = self.fc4batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc5(preds)
        return preds
