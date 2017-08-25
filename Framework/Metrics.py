"""
Define custom metrics here
"""

from sklearn.metrics import accuracy_score
from Helpers.TFUtilities import one_hot_to_binary
import numpy as np

def categorical_accuracy_score(y_true, y_pred):
    if np.ndim(y_true) == 2:
        y_true = one_hot_to_binary(y_true)
    if np.ndim(y_pred) == 2:
        y_pred = one_hot_to_binary(y_pred)

    return accuracy_score(y_true, y_pred)