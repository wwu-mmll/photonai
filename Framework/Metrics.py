"""
Define custom metrics here
"""

import numpy as np
from sklearn.metrics import accuracy_score

from Helpers.TFUtilities import one_hot_to_binary


def categorical_accuracy_score(y_true, y_pred):
    if np.ndim(y_true) == 2:
        y_true = one_hot_to_binary(y_true)
    if np.ndim(y_pred) == 2:
        y_pred = one_hot_to_binary(y_pred)

    return accuracy_score(y_true, y_pred)

def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

def variance_explained(y_true, y_pred):
    return np.square(pearson_correlation(y_true, y_pred))


