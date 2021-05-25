import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import ToPILImage


def transform_test(img):
    transformation = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    # print(np.array(transformation(np.array(Image.fromarray(np.array(img))))).shape)
    return transformation(np.array(Image.fromarray(np.array(img))))
