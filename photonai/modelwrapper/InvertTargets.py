from sklearn.base import BaseEstimator


class InvertTargets(BaseEstimator):

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        return 1 - y
