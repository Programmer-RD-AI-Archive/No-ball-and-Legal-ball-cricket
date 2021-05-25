import torch.nn.functional as F
import torch
import torch.nn as nn
from config import *


class Model_Clf_Testing_Params(nn.Module):
    def __init__(
        self,
        activation=F.relu,
        fc1_output=64,
        fc3_output=128,
        fc4_output=256,
        num_of_layers=2,
    ):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(
            3*112*112,
            fc1_output,
        )
        self.num_of_layers = num_of_layers
        self.fc1batchnorm = nn.BatchNorm1d(fc1_output)
        self.fc2 = nn.Linear(fc1_output, fc1_output)
        self.fc2batchnorm = nn.BatchNorm1d(fc1_output)
        self.fc3 = nn.Linear(fc1_output, fc3_output)
        self.fc3batchnorm = nn.BatchNorm1d(fc3_output)
        self.fc4 = nn.Linear(fc3_output, fc4_output)
        self.fc4batchnorm = nn.BatchNorm1d(fc4_output)
        self.fc6 = nn.Linear(fc4_output, fc3_output)
        self.fc6batchnorm = nn.BatchNorm1d(fc3_output)
        self.fc5 = nn.Linear(fc3_output, config["testing"]["NUM_CLASS"])

    def forward(self, X):
        X = X.view(
            -1,
            3*112*112,
        )
        preds = self.fc1(X)
        preds = self.fc1batchnorm(preds)
        preds = self.activation(preds)
        for _ in range(self.num_of_layers):
            preds = self.fc2(preds)
            preds = self.fc2batchnorm(preds)
            preds = self.activation(preds)
        preds = self.fc3(preds)
        preds = self.fc3batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc4(preds)
        preds = self.fc4batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc6(preds)
        preds = self.fc6batchnorm(preds)
        preds = self.activation(preds)
        preds = self.fc5(preds)
        return preds
