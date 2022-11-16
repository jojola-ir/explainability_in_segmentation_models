import os
from os.path import join

import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

def feature_maps(image, image_name, directory, layers):
    outputs = []
    names = []
    for layer in layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)

    if not os.path.exists("vgg16/"):
        os.makedirs("vgg16/")

    plt.savefig(str('results/{}/{}'.format(join("vgg16/", directory), image_name)), bbox_inches='tight')
    plt.close()