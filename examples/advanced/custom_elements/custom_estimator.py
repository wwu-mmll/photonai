import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, param1=0, param2=None):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None, **kwargs):
        """
        Adjust the underlying model or method to the data.

        Returns
        -------
        IMPORTANT: must return self!
        """
        return self

    def predict(self, X):
        """
        Use the learned model to make predictions.
        """
        return np.random.randint(0, 2, X.shape[0])
