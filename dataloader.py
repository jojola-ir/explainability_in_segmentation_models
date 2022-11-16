import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image


IMG_SIZE = 224


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image / 255.
    image = (image - 0.5) / 0.5  # normalize to [-1, 1]
    image = image.transpose(2, 0, 1)  # to c, h, w
    image = torch.from_numpy(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # create a mini-batch as expected by the model
    return image


def transformations():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform