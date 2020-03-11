import os
import numpy as np

from keras.models import load_model
from .load_images import load_single_image

def predict(path_to_image, model_location):

    model = load_model(model_location)
    img_to_predict = load_single_image(path_to_image, apply_contrast_stretching=True)
    img_to_predict = img_to_predict.reshape(1, img_to_predict.shape[0], img_to_predict.shape[1], img_to_predict.shape[2])

    prediction = model.predict(img_to_predict)
    prediction_num = np.argmax(prediction, axis=1)

    class_map = {0:"FULL_VISIBILITY", 1:"NO_VISIBILITY", 2:"PARTIAL_VISIBILITY"}
    predicted_class = class_map[prediction_num[0]]
    print("RESULT: predicted class is %s" % predicted_class )

    return predicted_class

