import cv2
import numpy as np
import tensorflow as tf
from starmart.input import Input, ImageInput
from starmart.result import Result, CompositeResult, NamedResult, ImageResult, ClassificationResult, Classification
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

model = None


def pre_start() -> None:
    global model
    model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')


def infer(data: Input) -> Result:
    # preprocessing image
    img = cv2.resize(data, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # classifying the image
    predictions = model.predict(img)
    # decoding model results (with built in tensorflow function)
    decoded_predictions = decode_predictions(predictions, top=10)[0]

    # parsing the results in order to match the output format
    classifications = []
    for prediction in decoded_predictions:
        classifications.append(Classification(label=prediction[1],
                                              confidence=prediction[2]))

    return ClassificationResult(classifications)


def input_format() -> Input:
    return ImageInput(None)


def output_format() -> Result:
    return CompositeResult([
        NamedResult('image', ImageResult(None)),
        NamedResult('classification_result', ClassificationResult(None))
    ])
