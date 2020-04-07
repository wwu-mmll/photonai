from sklearn.base import BaseEstimator, TransformerMixin


class InvertTargets(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        return 1 - y

    def fit_transform(self, X, y):
        return self.transform(X, y)

