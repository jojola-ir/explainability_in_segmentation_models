import os
from os.path import join

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import torchvision
from torchvision import models
from torchvis.selflib import util

import timm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataloader import preprocess_image, recreate_image, transformations



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
    # Get the index corresponding to the maximum score and the maximum score it
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


def activation_maximization_(model, selected_filter, model_name):
    res_path = join("results/", model_name)

    transform = transformations()

    random_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
    image = Image.fromarray(random_image)
    image = transform(image)
    image = Variable(image, requires_grad=True)

    # Define the optimizer
    optimizer = optim.Adam([image], lr=0.1)

    if model_name == "vgg16":
        length = len(model.features)
        layers_enum = model.features
        selected_layers = [0, 10, 24, 28]
    elif model_name == "ViT":
        length = len(model.blocks)
        layers_enum = model.blocks
        selected_layers = [0, 2, 5, 11]

    for selected_layer in selected_layers:
        for i in range(0, length):
            optimizer.zero_grad()

            x = image
            for idx, layer in enumerate(layers_enum):
                print(x.shape)
                x = layer(x)

                if idx == selected_layer:
                    break

            output = x[0, selected_filter]
            loss = -torch.mean(output)
            print("Iteration: {}, Loss: {}".format(i, loss.item()))

            loss.backward()
            optimizer.step()
            created_image = image.data.cpu().numpy().squeeze()

            if i % 5 == 0:
                plt.figure(figsize=(50,50))
                plt.imshow(created_image)
                plt.axis("off")

                save_dir = join(join(res_path, "activation_maximization"), "layer_{}".format(selected_layer))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                plt.savefig(str("{}/{}_{}_{}".format(save_dir, selected_layer, i)),
                            bbox_inches='tight')
                plt.close()


def activation_maximization(model, selected_filter, model_name):
    res_path = join("results/", model_name)

    transform = transformations()

    random_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
    image = Image.fromarray(random_image)
    image = transform(image)
    image = image.unsqueeze(0)
    image = Variable(image, requires_grad=True)

    optimizer = optim.Adam([image], lr=0.1)
    epochs = 100

    if model_name == "vgg16":
        layers_enum = model.features
        selected_layers = [0, 10, 24, 28]
    elif model_name == "ViT":
        layers_enum = model.blocks
        selected_layers = [0, 2, 5, 11]

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))
        for epoch in range(epochs):
            optimizer.zero_grad()

            x = image
            for idx, layer in enumerate(layers_enum):
                x = layer(x)

                if idx == selected_layer:
                    break

            output = x[0, selected_filter]
            loss = -torch.mean(output)
            print("Iteration: {}, Loss: {}".format(epoch, loss.item()))

            loss.backward()
            optimizer.step()
            """created_image = image.data.to("cpu").numpy()
            created_image = np.uint8(created_image).transpose(1, 2, 0)"""
            created_image = recreate_image(image)

            if epoch % 5 == 0:
                plt.figure(figsize=(50,50))
                plt.imshow(created_image)
                plt.axis("off")

                save_dir = join(join(res_path, "activation_maximization"), "layer_{}".format(selected_layer))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                plt.savefig(str("{}/{}_{}".format(save_dir, selected_layer, epoch)),
                            bbox_inches='tight')
                plt.close()