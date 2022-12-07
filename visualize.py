import os
import glob

from natsort import natsorted
from os.path import join
from PIL import Image


def make_gif(directory):
    splitted_dir = directory.split("/")
    model_name = splitted_dir[-3]
    extensions = ("JPG", "JPEG", "PNG", "jpg", "jpeg", "png")

    paths = []
    for root, dir, files in os.walk(directory):
        for f in files:
            ext = f.split(".")[-1]
            if ext in extensions:
                path = join(root, f)
                paths.append(path)

    paths = natsorted(paths)
    frames = [Image.open(path) for path in paths]
    size = (256, 256)
    for frame in frames:
        frame.thumbnail(size, Image.Resampling.LANCZOS)
    frame_one = frames[0]
    frame_one.save("{}/{}_{}.gif".format(directory, model_name, size[0]), format="GIF", append_images=frames,
               save_all=True, duration=1, loop=0, optimize=True, quality=10)


if __name__ == "__main__":
    make_gif("results/vgg16/feature_inversion/")