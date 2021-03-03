# we use BaseEstimator as to prepare the transformer for hyperparameter optimization
# we inherit the get_params and set_params methods
from sklearn.base import BaseEstimator


class CustomTransformer(BaseEstimator):

    def __init__(self, param1=0, param2=None):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.param1 = param1
        self.param2 = param2

    def fit(self, data, targets=None, **kwargs):
        """
        Adjust the underlying model or method to the data.

        Returns
        -------
        IMPORTANT: must return self!
        """
        return self

    def transform(self, data, targets=None, **kwargs):
        """
        Apply the method's logic to the data.
        """
        return data
