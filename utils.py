import os
from os.path import join

import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np

import matplotlib.pyplot as plt



def feature_maps(model, image, image_name, directory, model_name):
    img_activations_dir = join(image_name.split(".")[0])
    res_path = join("results/", model_name)

    activations = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    _ = model(image)

    if model_name == "vgg16":
        h1 = model.features[0].register_forward_hook(getActivation('conv2d_0'))
        h2 = model.features[10].register_forward_hook(getActivation('conv2d_10'))
        h3 = model.features[24].register_forward_hook(getActivation('conv2d_24'))
        h4 = model.features[28].register_forward_hook(getActivation('conv2d_28'))

        f_list = ['conv2d_0', 'conv2d_10', 'conv2d_24', 'conv2d_28']

        for f in f_list:
            plt.figure(figsize=(10, 10))
            for i in range(8):
                for j in range(8):
                    a = plt.subplot(8, 8, 8 * i + j + 1)
                    imgplot = plt.imshow(activations[f][0, 8 * i + j])
                    a.axis("off")
            plt.savefig(str("{}/{}".format(join(res_path, directory), img_activations_dir)),
                        bbox_inches='tight')
            plt.close()

    elif model_name == "ViT":
        h1 = model.blocks[0].register_forward_hook(getActivation('Block_0'))
        h2 = model.blocks[2].register_forward_hook(getActivation('Block_2'))
        h3 = model.blocks[5].register_forward_hook(getActivation('Block_5'))
        h4 = model.blocks[11].register_forward_hook(getActivation('Block_11'))

        f_list = ['Block_0', 'Block_2', 'Block_5', 'Block_11']

        plt.figure(figsize=(30, 50))
        for i in range(8):
            for j in range(8):
                a = plt.subplot(8, 8, 8 * i + j + 1)
                to_show = activations['Block_0'][0, 1:, 8 * i + j].reshape((14, 14))
                to_show = cv2.resize(np.array(to_show), (224, 224))
                plt.xticks([]), plt.yticks([])
                imgplot = plt.imshow(to_show)
                a.axis("off")

        for f in f_list:
            plt.figure(figsize=(30, 50))
            for i in range(8):
                for j in range(8):
                    a = plt.subplot(8, 8, 8 * i + j + 1)
                    to_show = activations[f][0, 1:, 8 * i + j].reshape((14, 14))
                    to_show = cv2.resize(np.array(to_show), (224, 224))
                    plt.xticks([]), plt.yticks([])
                    imgplot = plt.imshow(to_show)

            plt.savefig(str("{}/{}/{}".format(join(res_path, directory), img_activations_dir, image_name)),
                        bbox_inches='tight')
            plt.close()