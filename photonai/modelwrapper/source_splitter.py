from sklearn.base import BaseEstimator, TransformerMixin


class SourceSplitter(BaseEstimator, TransformerMixin):
    """
    SourceSplitter transforms data by reducing to data[:, column_indices].
    """

    def __init__(self, column_indices: list):
        self.column_indices = column_indices

    def fit(self, X, y=None, **kwargs):
        return

    def transform(self, X):
        return X[:,self.column_indices]

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)