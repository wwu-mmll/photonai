from sklearn.base import BaseEstimator, RegressorMixin
from typing import Union


class RangeRestrictor(BaseEstimator, RegressorMixin):

    def __init__(self, low: Union[int, float] = 0, high: Union[int, float] = 100):
        self.low = low
        self.high = high
        self.needs_y = False
        self.needs_covariates = True

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, **kwargs):
        X[X > self.high] = self.high
        X[X < self.low] = self.low
        return X
