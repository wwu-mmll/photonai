from photonai.base.PhotonBase import PipelineElement
import importlib
from photonai.photonlogger.Logger import Logger


class AnomalyDetectorWrapper:
    def __init__(self, **kwargs: dict):
        self.estimator = None
        self.estimator_name = None
        self.init_wrapper_class(**kwargs)

    def fit(self, X, y=None):
        self.estimator.fit(X[y == 1])
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def set_params(self, **params):
        self.init_wrapper_class(**params)
        del params['wrapped_estimator']
        self.estimator.set_params(**params)
        return self

    def get_params(self, deep=True):
        if self.estimator is None:
            return {}
        params = self.estimator.get_params(deep)
        params['wrapped_estimator'] = self.estimator_name
        return params

    def init_wrapper_class(self, **kwargs):
        if 'wrapped_estimator' in kwargs:
            if self.estimator_name != kwargs['wrapped_estimator']:
                self.estimator_name = kwargs['wrapped_estimator']
                del kwargs['wrapped_estimator']
                try:
                    desired_class_info = PipelineElement.ELEMENT_DICTIONARY[self.estimator_name]
                    desired_class_home = desired_class_info[0]
                    desired_class_name = desired_class_info[1]
                    imported_module = importlib.import_module(desired_class_home)
                    desired_class = getattr(imported_module, desired_class_name)
                    self.estimator = desired_class(**kwargs)

                except AttributeError as ae:
                    Logger().error('ValueError: Could not find according class:'
                                   + str(PipelineElement.ELEMENT_DICTIONARY[self.estimator_name]))