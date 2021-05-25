from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from config import *


def tl_model(device):
    model = models.resnet18(pretrained=True).to(device)
    model = model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model
