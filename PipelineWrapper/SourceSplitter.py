from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SourceSplitter:
    """
    Source splitter
    """

    def __init__(self, column_indices: np.array):
        self.column_indices = column_indices

    def fit(self, X, y):
        return self.transform(X)

    def transform(self, X):
        return X[:,self.column_indices]

    def predict(self, X):
        return self.transform(X)

