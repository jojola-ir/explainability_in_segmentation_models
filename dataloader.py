import copy

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
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im