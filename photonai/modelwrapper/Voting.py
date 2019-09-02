from sklearn.base import BaseEstimator
import numpy as np
from scipy import stats


class Voting(BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, strategy='mean'):

        self.STRATEGY_DICT = {'mean': np.mean,
                              'median': np.median,
                              'most_frequent': Voting._most_frequent,
                              'min': np.min,
                              'max': np.max}

        self._strategy = None
        self.strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_val):
        if strategy_val not in self.STRATEGY_DICT:
            raise ValueError("Strategy " + str(self.strategy) + " is not supported right now. ")
        else:
            self._strategy = strategy_val

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, **kwargs):
        if X is not None:
            output = self.STRATEGY_DICT[self.strategy](X, axis=1)
            if isinstance(X[0], int):
                return [np.round(output)]
            else:
                return output

    @staticmethod
    def _most_frequent(X, axis):
        mode_obj = stats.mode(X, axis=axis)
        return [i[0] for i in mode_obj.mode]





