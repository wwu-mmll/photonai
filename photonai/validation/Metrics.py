"""
Define custom metrics here
The method stub of all metrics is
function_name(y_true, y_pred)
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.metrics import categorical_crossentropy as keras_categorical_crossentropy
from keras import backend as K
from ..helpers.TFUtilities import one_hot_to_binary, binary_to_one_hot


def categorical_accuracy_score(y_true, y_pred):
    if np.ndim(y_true) == 2:
        y_true = one_hot_to_binary(y_true)
    if np.ndim(y_pred) == 2:
        y_pred = one_hot_to_binary(y_pred)

    return accuracy_score(y_true, y_pred)

def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

def variance_explained_score(y_true, y_pred):
    return np.square(pearson_correlation(y_true, y_pred))

# def categorical_cross_entropy(y_true, y_pred):
#     return categorical_crossentropy(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))

# def categorical_crossentropy(y_true, y_pred):
#     return K.eval(keras_categorical_crossentropy(K.variable(binary_to_one_hot(y_true), 'float64'),
#                                                  K.variable(binary_to_one_hot(y_pred), 'float64')))
