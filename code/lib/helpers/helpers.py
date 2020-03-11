import math
import numpy as np
import pandas as pd
from skimage import exposure


def to_one_channel(img):
    """ Convert an RGB image to one channel image.

    :param img: numpy nd-array - input image.

    :return: numpy nd-array - output image.
    """
    return img[:, :, 2]


def contrast_stretching(img):
    """ Add contrast stretching to an image to aid the model.

    :param img: numpy nd-array - input image.

    :return: numpy nd-array - output image after contrast stretching.
    """
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


def oversample_minority_classes(df, target):
    """ Oversample the minority classes to match
    the majority class.

    :param df: pandas dataframe - input df.
    :param target: string - classification target column.

    :return: pandas datframe - oversampled version
    """

    class_count = df[target].value_counts()

    print("Before oversampling: %s" % class_count)
    for i in range(1, len(class_count)):
        df_i = df[df[target] == i]
        oversampling_factor_i = class_count[0] / float(class_count[i])
        print(len(df_i))
        print("Oversampling factor for class %i: %s" % (i, str(oversampling_factor_i)))

        # Integer part of oversampling
        df = df.append(
            [df_i] * int(math.floor(oversampling_factor_i) - 1),
            ignore_index=False)
        # Float part of oversampling
        df = df.append(
            [df_i.sample(frac=oversampling_factor_i % 1)],
            ignore_index=False)

    print("After oversampling: %s" % df[target].value_counts())
    print("Shape after oversampling: %s" % str(df.shape))

    return df
