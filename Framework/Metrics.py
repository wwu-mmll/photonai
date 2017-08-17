"""
Define custom metrics here
"""

from sklearn.metrics import accuracy_score
from Helpers.TFUtilities import oneHot

def categorical_accuracy_score(y_true, y_pred):
    return accuracy_score(oneHot(y_true, reverse=True), oneHot(y_pred, reverse=True))