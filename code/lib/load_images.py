import os
import numpy as np
import matplotlib.pyplot as plt
from .helpers import to_one_channel, contrast_stretching

def load_images(img_dir, file_names, apply_contrast_stretching=True):
    """Loads images from file location list.

    :param img_dir: atr - directory containing images.
    :param file_names: pd.Series - list of file names for each image.

    :return: numpy array with images
    """
    print("loading %s images" % len(file_names))
    X_images = []
    for file in file_names:
        img = load_single_image(os.path.join(img_dir, file), apply_contrast_stretching)
        X_images.append(img)
    X_images = np.array(X_images)

    return X_images

def load_single_image(path_to_img, apply_contrast_stretching=True):
    """Loads an image from indicated path.

    :param path_to_img: str - path to image.

    :return: numpy array
    """
    img = to_one_channel(plt.imread(path_to_img))
    img = np.stack((img,) * 3, axis=-1)

    if apply_contrast_stretching:
        img = contrast_stretching(img)

    return img