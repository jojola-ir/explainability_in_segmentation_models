import os
from os.path import join

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
from torchvision import models
from torchvis.selflib import util

import timm

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataloader import transformations



def feature_maps(model, image, image_name, directory, model_name, data_name):
    img_activations_dir = join(image_name.split(".")[0])
    res_path = join(join("results/", model_name), data_name)

    activations = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    if model_name == "vgg16":
        h1 = model.features[0].register_forward_hook(getActivation('conv2d_0'))
        h2 = model.features[10].register_forward_hook(getActivation('conv2d_10'))
        h3 = model.features[24].register_forward_hook(getActivation('conv2d_24'))
        h4 = model.features[28].register_forward_hook(getActivation('conv2d_28'))

        _ = model(image)

        for elem in activations:
            activations[elem] = activations[elem].to("cpu").numpy()

        f_list = ['conv2d_0', 'conv2d_10', 'conv2d_24', 'conv2d_28']

        for f in f_list:
            print(f)
            plt.figure(figsize=(10, 10))
            for i in range(8):
                for j in range(8):
                    a = plt.subplot(8, 8, 8 * i + j + 1)
                    imgplot = plt.imshow(activations[f][0, 8 * i + j])
                    a.axis("off")

            save_dir = join(join(join(res_path, "feature_maps"), directory), img_activations_dir)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fname = f + '_' + image_name
            plt.savefig(str("{}/{}".format(save_dir, fname)),
                        bbox_inches='tight')
            plt.close()

    elif model_name == "ViT":
        h1 = model.blocks[0].register_forward_hook(getActivation('Block_0'))
        h2 = model.blocks[2].register_forward_hook(getActivation('Block_2'))
        h3 = model.blocks[5].register_forward_hook(getActivation('Block_5'))
        h4 = model.blocks[11].register_forward_hook(getActivation('Block_11'))

        _ = model(image)

        for elem in activations:
            activations[elem] = activations[elem].to("cpu").numpy()

        f_list = ['Block_0', 'Block_2', 'Block_5', 'Block_11']

        for f in f_list:
            print(f)
            plt.figure(figsize=(30, 50))
            for i in range(8):
                for j in range(8):
                    a = plt.subplot(8, 8, 8 * i + j + 1)
                    to_show = activations[f][0, 1:, 8 * i + j].reshape((14, 14))
                    to_show = cv2.resize(np.array(to_show), (224, 224))
                    plt.xticks([]), plt.yticks([])
                    imgplot = plt.imshow(to_show)
                    a.axis("off")

            save_dir = join(join(join(res_path, "feature_maps"), directory), img_activations_dir)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fname = f + '_' + image_name
            plt.savefig(str("{}/{}".format(save_dir, fname)),
                        bbox_inches='tight')
            plt.close()


def saliency_maps(model, image, image_name, directory, model_name, data_name):
    img_activations_dir = join(image_name.split(".")[0])
    res_path = join(join("results/", model_name), data_name)

    image = Variable(image, requires_grad=True)

    model.eval()

    scores = model(image)
    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    '''
    backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
    score_max with respect to nodes in the computation graph
    '''
    score_max.backward()
    '''
    Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
    R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
    across all colour channels.
    '''
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency = saliency.to("cpu").numpy()

    plt.figure(figsize=(15,15))
    plt.imshow(saliency[0], cmap='hot')
    plt.axis("off")

    save_dir = join(join(join(res_path, "saliency_maps"), directory), img_activations_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(str("{}/{}".format(save_dir, image_name)),
                bbox_inches='tight')
    plt.close()