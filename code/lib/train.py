import pandas as pd
import os
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from code.lib.load_images import load_images
from code.lib.helpers import oversample_minority_classes
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .load_images import load_images
from .helpers import oversample_minority_classes
from .model import get_model


def train_model(img_dir, metadata_dir, out_dir):
    # load meta data
    df_meta = pd.read_csv(os.path.join(metadata_dir, "gicsd_labels.csv"))
    # format the meta data
    df_meta.columns = df_meta.columns.str.strip()
    df_meta.LABEL = df_meta.LABEL.str.strip()
    df_meta['BACKGROUND_ID'] = df_meta['IMAGE_FILENAME'].apply(lambda x: x.split("_")[2]).apply(lambda x: int(x))
    df_meta['CARD_ID'] = df_meta['IMAGE_FILENAME'].apply(lambda x: x.split("_")[1]).apply(lambda x: int(x))
    df_meta['IMAGE_ID'] = df_meta['IMAGE_FILENAME'].apply(lambda x: x.split("_")[3]).apply(lambda x: x.split(".")[0]).apply(
        lambda x: int(x))

    # convert labels to numeric categories
    le = preprocessing.LabelEncoder()
    le.fit(df_meta['LABEL'])
    print(list(le.classes_))
    df_meta['TARGET'] = le.transform(df_meta['LABEL'])

    # oversampling minority classes to deal with class imbalance
    df_train_oversampled = oversample_minority_classes(df_meta, "TARGET")

    # get oversampled images and corresponding labels
    x_train_oversampled = df_train_oversampled["IMAGE_FILENAME"]
    y_train_oversampled = df_train_oversampled["TARGET"]

    model = get_model()
    model.summary()

    # load the training data
    X_train_images = load_images(img_dir=img_dir, file_names=x_train_oversampled, apply_contrast_stretching=True)
    # converting to one-hot-encoding required by categorical_crossentropy loss
    y_train_binary = to_categorical(y_train_oversampled)

    # create generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=4,
        width_shift_range=[-2, 2],
        height_shift_range=[-5, 5])

    # train model
    model.fit_generator(datagen.flow(X_train_images, y_train_binary, batch_size=32), verbose=1, epochs=35, shuffle=True)

    # save model
    model.save(os.path.join(out_dir, 'prod_model.h5'))
    print("model saved to %s " % os.path.join(out_dir, 'prod_model.h5'))
