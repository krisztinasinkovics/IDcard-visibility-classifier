import keras
from keras.models import Sequential, Model
from keras.layers import Concatenate, Activation, Dropout, Flatten, Dense, BatchNormalization

from keras.applications.resnet50 import ResNet50


def get_model():
    # base ResNet50 layer using pre-trained weights to aid recognition of shapes and textures
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(192, 192, 3))

    add_model = Sequential()
    add_model.add(Flatten())
    add_model.add(Dense(256, activation='relu', input_dim=6 * 6 * 2048))
    add_model.add(Dropout(0.50))
    add_model.add(Dense(128, activation='relu'))
    add_model.add(Dropout(0.50))
    add_model.add(Dense(64, activation='relu'))
    add_model.add(Dense(3, activation='softmax'))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    print(base_model.output)
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', metrics=['acc'],
                  optimizer=adam)
    return model