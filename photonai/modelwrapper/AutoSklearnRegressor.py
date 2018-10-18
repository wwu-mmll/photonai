from autosklearn.regression import AutoSklearnRegressor as ASR
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class AutoSklearnRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, time_left_for_this_task=120, per_run_time_limit=30):
        self.model = ASR(per_run_time_limit=per_run_time_limit, time_left_for_this_task=time_left_for_this_task)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model = self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        return self.model.predict(X)
