import copy

import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

IMG_SIZE = 224


def load_image(image, device):
    transform = transformations()

    if type(image) == np.ndarray:
        image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    image = Variable(image, requires_grad=True)
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
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    return recreated_im
