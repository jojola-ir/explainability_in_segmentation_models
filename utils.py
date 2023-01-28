import copy
import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import optim
from torch.autograd import Variable

from dataloader import load_image, recreate_image


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
            image = load_image(random_image, device)
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


def get_output_from_layer(x, model, selected_layer):
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx == selected_layer:
            break
    return x


def vgg_feature_inversion(image, image_name, model, selected_layers, selected_filters, device):
    res_path = join("results/", "vgg16")

    def alpha_norm(input_matrix, alpha):
        """Converts matrix to vector then calculates the alpha norm."""
        alpha_norm = ((input_matrix.view(-1)) ** alpha).sum()
        return alpha_norm

    def total_variation_norm(input_matrix, beta):
        """Total variation norm is the second norm in the paper
            represented as R_V(x)."""
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom) ** 2 +
                            (to_check - one_right) ** 2) ** (beta / 2)).sum()
        return total_variation

    def euclidian_loss(org_matrix, target_matrix):
        """Euclidian loss is the main loss function in the paper
        ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2.
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    image = load_image(image, device)

    epochs = 200

    for selected_layer in selected_layers:
        print("\nProcessing layer: {}".format(selected_layer))
        vgg = model.features
        vgg = vgg[:selected_layer + 1]
        vgg = vgg.to(device)

        for selected_filter in selected_filters:
            print("Filter: {}".format(selected_filter))
            random_image = np.uint8(255 * np.random.normal(0, 1, (224, 224, 3)))
            random_image = load_image(random_image, device)
            optimizer = optim.SGD([random_image], lr=1e4, momentum=0.9)

            target = get_output_from_layer(image, vgg, selected_layer)

            # Alpha regularization parameters
            # Parameter alpha, which is actually sixth norm
            alpha_reg_alpha = 6
            # The multiplier, lambda alpha
            alpha_reg_lambda = 1e-7

            # Total variation regularization parameters
            # Parameter beta, which is actually second norm
            tv_reg_beta = 2
            # The multiplier, lambda beta
            tv_reg_lambda = 1e-8

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()

                output = get_output_from_layer(random_image, vgg, selected_layer)

                euc_loss = 1e-1 * euclidian_loss(target.detach(), output)
                # Calculate alpha regularization
                reg_alpha = alpha_reg_lambda * alpha_norm(random_image, alpha_reg_alpha)
                # Calculate total variation regularization
                reg_total_variation = tv_reg_lambda * total_variation_norm(random_image, tv_reg_beta)

                loss = euc_loss + reg_alpha + reg_total_variation

                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print("Epoch {} - Loss: {}".format(epoch, loss.item()))

                    created_image = recreate_image(random_image)

                    plt.figure(figsize=(30, 30))
                    img_plot = plt.imshow(created_image)
                    plt.axis("off")

                    save_dir = join(join(join(res_path, "feature_inversion"),
                                         "layer_{}".format(selected_layer)),
                                    "filter_{}".format(selected_filter))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    plt.savefig(str("{}/{}".format(save_dir, image_name)),
                                bbox_inches='tight')
                    plt.close()

                if epoch % 40 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 1 / 10


def vit_feature_inversion(image, image_name, model, selected_layers, selected_filters, device):
    res_path = join("results/", "ViT")

    def alpha_norm(input_matrix, alpha):
        """Converts matrix to vector then calculates the alpha norm."""
        alpha_norm = ((input_matrix.view(-1)) ** alpha).sum()
        return alpha_norm

    def total_variation_norm(input_matrix, beta):
        """Total variation norm is the second norm in the paper
            represented as R_V(x)."""
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom) ** 2 +
                            (to_check - one_right) ** 2) ** (beta / 2)).sum()
        return total_variation

    def euclidian_loss(org_matrix, target_matrix):
        """Euclidian loss is the main loss function in the paper
        ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2.
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

    image = load_image(image, device)

    epochs = 200

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
            print("Filter: {}".format(selected_filter))
            random_image = np.uint8(255 * np.random.normal(0, 1, (224, 224, 3)))
            random_image = load_image(random_image, device)
            optimizer = optim.SGD([random_image], lr=1e4, momentum=0.9)

            target = vit(image)

            # Alpha regularization parameters
            # Parameter alpha, which is actually sixth norm
            alpha_reg_alpha = 6
            # The multiplier, lambda alpha
            alpha_reg_lambda = 1e-7

            # Total variation regularization parameters
            # Parameter beta, which is actually second norm
            tv_reg_beta = 2
            # The multiplier, lambda beta
            tv_reg_lambda = 1e-8

            for epoch in range(1, epochs + 1):
                optimizer.zero_grad()

                output = vit(random_image)

                euc_loss = 1e-1 * euclidian_loss(target.detach(), output)
                # Calculate alpha regularization
                reg_alpha = alpha_reg_lambda * alpha_norm(random_image, alpha_reg_alpha)
                # Calculate total variation regularization
                reg_total_variation = tv_reg_lambda * total_variation_norm(random_image, tv_reg_beta)

                loss = euc_loss + reg_alpha + reg_total_variation

                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print("Epoch {} - Loss: {}".format(epoch, loss.item()))

                    created_image = recreate_image(random_image)

                    plt.figure(figsize=(30, 30))
                    img_plot = plt.imshow(created_image)
                    plt.axis("off")

                    save_dir = join(join(join(res_path, "feature_inversion"),
                                         "layer_{}".format(selected_layer)),
                                    "filter_{}".format(selected_filter))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    plt.savefig(str("{}/{}".format(save_dir, image_name)),
                                bbox_inches='tight')
                    plt.close()

                if epoch % 40 == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 1 / 10