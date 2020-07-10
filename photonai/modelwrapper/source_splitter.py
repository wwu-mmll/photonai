from sklearn.base import BaseEstimator, TransformerMixin


class SourceSplitter(BaseEstimator, TransformerMixin):
    """
    Source splitter
    """

    def __init__(self, column_indices: list):
        self.column_indices = column_indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, self.column_indices]

