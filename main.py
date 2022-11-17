import os
from os.path import join

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import timm
from PIL import Image

from dataloader import load_image, transformations
from models import VisionTransformer
from utils import feature_maps
from weights import load_weights


def model_builder(model_name):
    if model_name == "ViT":
        model = load_weights("vit_base_patch16_224", save=False)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)

    return model


def architecture(model):
    weights = []
    layers = []

    model_children = list(model.children())

    for child in model_children:
        if type(child) == nn.Conv2d:
            weights.append(child.weight)
            layers.append(child)
        elif type(child) == nn.Sequential:
            for c in child:
                if type(c) == nn.Conv2d:
                    weights.append(c.weight)
                    layers.append(c)

    count = len(layers)

    return weights, layers, count


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_name = "vgg16"
    model = model_builder(model_name)
    weights, layers, count = architecture(model)

    model = model.to(device)

    data_path = "../datasets/birds/"
    results_path = "results/"

    transform = transformations()

    image_path = join(data_path)

    for root, dir, files in os.walk(data_path):

        for f in files:
            if f.endswith(".jpg"):
                image = Image.open(join(root, f))

                image = transform(image)
                image = image.unsqueeze(0)
                image = image.to(device)

                d = root.split("/")[-1]
                print("\nProcessing {}/{}".format(d, f))

                feature_maps(model, image, f, d, model_name)