"""
Define custom metrics here
The method stub of all metrics is
function_name(y_true, y_pred)
"""

import numpy as np
from ..photonlogger.Logger import Logger

def categorical_accuracy_score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    from ..helpers.TFUtilities import one_hot_to_binary, binary_to_one_hot
    if np.ndim(y_true) == 2:
        y_true = one_hot_to_binary(y_true)
    if np.ndim(y_pred) == 2:
        y_pred = one_hot_to_binary(y_pred)

    return accuracy_score(y_true, y_pred)


def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]


def variance_explained_score(y_true, y_pred):
    return np.square(pearson_correlation(y_true, y_pred))


def sensitivity(y_true, y_pred):  # = true positive rate, hit rate, recall
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    else:
        Logger().info('Sensitivity (metric) is valid only for binary classification problems. You have ' + str(len(np.unique(y_true))) + ' classes.')
        return np.nan


def specificity(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        Logger().info('Specificity (metric) is valid only for binary classification problems. You have ' + str(len(np.unique(y_true))) + ' classes.')
        return np.nan

def balanced_accuracy(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        return (specificity(y_true, y_pred) + sensitivity(y_true, y_pred)) / 2
    else:
        Logger().info('Specificity (metric) is valid only for binary classification problems. You have ' + str(len(np.unique(y_true))) + ' classes.')
        return np.nan


# def categorical_cross_entropy(y_true, y_pred):
#     import tensorflow as tf
#     from keras.metrics import categorical_crossentropy as keras_categorical_crossentropy
#     return categorical_crossentropy(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred))


# def categorical_crossentropy(y_true, y_pred):
#     from keras.metrics import categorical_crossentropy as keras_categorical_crossentropy
#     from keras import backend as K
#     return K.eval(keras_categorical_crossentropy(K.variable(binary_to_one_hot(y_true), 'float64'),
#                                                  K.variable(binary_to_one_hot(y_pred), 'float64')))
