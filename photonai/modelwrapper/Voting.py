from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np
from scipy import stats


class PhotonVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):

        self.strategy = PhotonVotingClassifier._most_frequent

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, **kwargs):
        if X is not None:
            output = self.strategy(X, axis=1)
            if isinstance(X[0], int):
                return [np.round(output)]
            else:
                return output

    @staticmethod
    def _most_frequent(X, axis):
        mode_obj = stats.mode(X, axis=axis)
        return [i[0] for i in mode_obj.mode]


class PhotonVotingRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, strategy='mean'):

        self.STRATEGY_DICT = {'mean': np.mean,
                              'median': np.median}

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
