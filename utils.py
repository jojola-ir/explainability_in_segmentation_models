import copy
import os
from os.path import join

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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

from dataloader import load_image, recreate_image, transformations
from models import build_fc_reconstruction_model


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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


def activation_maximization(model, model_name):
    res_path = join("results/", model_name)

    transform = transformations()

    random_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
    image = Image.fromarray(random_image)
    image = transform(image)
    image = image.unsqueeze(0)
    image = Variable(image, requires_grad=True)

    optimizer = optim.Adam([image], lr=0.1)
    epochs = 100

    if model_name == "ViT":
        embedding = model.patch_embed
        image = embedding(image)

    if model_name == "vgg16":
        layers_enum = model.features
        selected_layers = [0, 10, 24, 28]
    elif model_name == "ViT":
        layers_enum = model.blocks
        selected_layers = [0, 2, 5, 11]

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))
        for selected_filter in range(0, 64, 16):
            print("\nProcessing filter: {}".format(selected_filter))
            for epoch in range(1, epochs + 1):
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
                created_image = recreate_image(image, model_name)

                if epoch % 5 == 0:
                    plt.figure(figsize=(30,30))
                    img_plot = plt.imshow(created_image)
                    plt.axis("off")

                    save_dir = join(join(join(res_path, "activation_maximization"),
                                         "layer_{}".format(selected_layer)),
                                    "filter_{}".format(selected_filter))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    plt.savefig(str("{}/{}_{}".format(save_dir,selected_layer, epoch)),
                                bbox_inches='tight')
                    plt.close()


def vgg_activation_maximization(model, selected_layers, selected_filters, device):
    res_path = join("results/", "vgg16")

    epochs = 100

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))
        vgg = model.features
        vgg = vgg[:selected_layer + 1]
        vgg = vgg.to(device)
        for selected_filter in selected_filters:
            random_image = np.uint8(255 * np.random.normal(0, 1, (224, 224, 3)))
            image = load_image(random_image, device)
            optimizer = optim.Adam([image], lr=0.1)

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()

                x = image
                for idx, layer in enumerate(vgg):
                    x = layer(x)

                    if idx == selected_layer:
                        break

                output = x[0, selected_filter]
                loss = -torch.mean(output)

                loss.backward()
                optimizer.step()

                created_image = recreate_image(image)

                if epoch == epochs:
                    plt.figure(figsize=(30, 30))
                    img_plot = plt.imshow(created_image)
                    plt.axis("off")

                    save_dir = join(join(res_path, "activation_maximization"),
                                    "layer_{}".format(selected_layer))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    plt.savefig(str("{}/{}_{}".format(save_dir, selected_layer, selected_filter)),
                                bbox_inches='tight')
                    plt.close()

            print("Layer {} / filter {} - Loss: {}".format(selected_layer, selected_filter, loss.item()))


def vit_activation_maximization(model, selected_layers, selected_filters, device):
    res_path = join("results/", "ViT")

    epochs = 100

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))
        vit = copy.deepcopy(model)
        for block in range(selected_layer, len(vit.blocks)):
            vit.blocks[block] = Identity()
        vit.norm = Identity()
        vit.fc_norm = Identity()
        vit.head = Identity()
        vit = vit.to(device)

        for selected_filter in selected_filters:
            random_image = np.uint8(255 * np.random.normal(0, 1, (224, 224, 3)))
            image = load_image(random_image)
            optimizer = optim.Adam([image], lr=0.1)

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()

                x = image

                output = vit(x)
                loss = -torch.mean(output[0, :, selected_filter])

                loss.backward()
                optimizer.step()

                created_image = recreate_image(image)

                if epoch == epochs:

                    plt.figure(figsize=(30, 30))
                    img_plot = plt.imshow(created_image)
                    plt.axis("off")

                    save_dir = join(join(res_path, "activation_maximization"),
                                    "layer_{}".format(selected_layer))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    plt.savefig(str("{}/{}_{}".format(save_dir, selected_layer, selected_filter)),
                                bbox_inches='tight')
                    plt.close()

            print("Layer {} / filter {} - Loss: {}".format(selected_layer, selected_filter, loss.item()))


def grad_cam(model, image, image_name, directory, model_name, data_name):
    img_activations_dir = join(image_name.split(".")[0])
    res_path = join(join("results/", model_name), data_name)

    if model_name == "vgg16":
        layers_enum = model.features
        selected_layers = [0, 10, 24, 28]
    elif model_name == "ViT":
        layers_enum = model.blocks
        selected_layers = [0, 2, 5, 11]

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))

        cam = GradCAM(model=model, target_layer=selected_layer, use_cuda=False)
        targets = [ClassifierOutputTarget(281)]

        grayscale_cam = cam(input_tensor=image, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)


def feature_inversion(image, model, model_name, device):
    res_path = join("results/", model_name)

    image = load_image(image, device)

    conv_net = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
    fc_net = build_fc_reconstruction_model(1000, 256)