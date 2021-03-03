from sklearn.base import BaseEstimator


class CustomCovariateTransformer(BaseEstimator):
    """
    Any algorithm that uses the target values to change the data (e.g. transform male and female differently?)
    or that changes the dataset including targets in total (e.g. sample pairing oversampling)
    """

    def __init__(self, param1=0, param2=None):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.param1 = param1
        self.param2 = param2

        self.needs_covariates = True

    def fit(self, data, targets=None, **kwargs):
        """
        Adjust the underlying model or method to the data.
        You can find any covariate data in the kwargs.

        E.g. if you did hyperpipe.fit(X, y, age=my_age_variable)
        you can access the matched age values to X and y by
        age = kwargs["age"]
        Returns
        -------
        IMPORTANT: must return self!
        """
        return self

    def transform(self, data, targets=None, **kwargs):
        """
        Apply the method's logic to the data using any covariate not included in the feature set X.

        E.g. if you did hyperpipe.fit(X, y, age=my_age_variable)
        you can access the matched age values to X and y by
        age = kwargs["age"]

        You can change the  values for later applied algorithms by modifying them and returning a new dict
        kwargs["age"] = new_age_variable

        IMPORTANT: MUST return both data and kwargs
        """
        return data, kwargs
