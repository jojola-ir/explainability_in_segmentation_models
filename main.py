from PIL.Image import Image

from models import model_builder
from utils import *


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

    print("Device : {}".format(device))

    model_name = "ViT"
    model = model_builder(model_name)
    weights, layers, count = architecture(model)

    model = model.to(device)

    data_path = "../datasets/one_shot/"
    data_name = data_path.split("/")[-2]
    results_path = "results/"

    image_path = join(data_path, "001.jpg")
    image = Image.open(image_path)
    #image = load_image(image, device)

    # activation maximization
    selected_layers = [0, 2, 5, 11]
    selected_filters = range(0, 64, 8)
    #vgg_feature_inversion(image, image_path.split("/")[-1], model, selected_layers, selected_filters, device)
    vit_feature_inversion(image, image_path.split("/")[-1], model, selected_layers, selected_filters, device)
    #vgg_activation_maximization(model, selected_layers, selected_filters, device)

    """for root, dir, files in os.walk(data_path):

        for f in files:
            if f.endswith(".jpg") or f.endswith(".JPEG"):
                image = Image.open(join(root, f))

                d = root.split("/")[-1]
                print("\nProcessing {}/{}".format(d, f))

                # activation maximization
                #activation_maximization(model, selected_filter, d, model_name, data_name)

                # feature inversion
                vgg_feature_inversion(image, f, model, selected_layers, selected_filters, device)"""

