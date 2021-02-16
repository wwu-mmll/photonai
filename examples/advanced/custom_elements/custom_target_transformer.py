from sklearn.base import BaseEstimator


class CustomYTransformer(BaseEstimator):
    """
    Any algorithm that uses the target values to change the data (e.g. transform male and female differently?)
    or that changes the dataset including targets in total (e.g. sample pairing oversampling)
    """

    def __init__(self, param1=0, param2=None):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.param1 = param1
        self.param2 = param2

        self.needs_y = True

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
        Apply the method's logic to the data and targets.
        IMPORTANT: MUST return data AND targets
        """
        return data, targets