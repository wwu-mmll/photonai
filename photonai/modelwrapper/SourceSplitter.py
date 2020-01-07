from sklearn.base import BaseEstimator


class SourceSplitter(BaseEstimator):
    """
    Source splitter
    """

    def __init__(self, column_indices: list):
        self.column_indices = column_indices

    def fit(self, X, y):
        return self.transform(X)

    def transform(self, X):
        X_split = X[:,self.column_indices]
        return X_split

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X):
        return self.transform(X)

