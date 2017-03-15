
from typing import Generic, TypeVar

from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from HPOFramework.HPOptimizers import GridSearchOptimizer
from TFLearnPipelineWrapper.TFDNNClassifier import TFDNNClassifier


class HyperpipeManager(BaseEstimator):

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer}

    def __init__(self, X, y=None, groups=None, config = None):

        # self.param_dict = param_dict
        self.pipeline_param_list = {}
        self.X = X
        self.y = y
        self.groups = groups
        # Todo: implement CV adjustment strategy
        self.cv = KFold(n_splits=5)
        self.cv_iter = list(self.cv.split(X, y, groups))
        self.pipeline_elements = []
        self.pipe = None

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

        print(len(self.pipeline_elements))

    def __iadd__(self, pipe_element):
        if isinstance(pipe_element, PipelineElement):
            self.pipeline_elements.append(pipe_element)
            return self
        else:
            #Todo: raise error
            raise TypeError("Element must be of type Pipeline Element")

    def optimize(self, optimization_strategy):

        # 0. build pipeline...
        pipeline_steps = []
        for item in self.pipeline_elements:
            pipeline_steps.append((item.name, item.base_element))
        self.pipe = Pipeline(pipeline_steps)

        # and bring hyperparameters into sklearn pipeline syntax
        self.organize_parameters()

        # 1. instantiate optimizer
        #  Todo: check if optimizer strategy is already implemented
        optimizer_class = self.OPTIMIZER_DICTIONARY[optimization_strategy]
        optimizer_instance = optimizer_class(self.pipeline_param_list)

        # 2. iterate next_config
        for specific_config in optimizer_instance.next_config:
            hp = Hyperpipe(self.pipe, specific_config)
            config_loss = hp.calculate_cv_loss(self.X, self.y, self.cv_iter)
            print(config_loss)

    def organize_parameters(self):
        pipeline_dict = dict()
        for item in self.pipeline_elements:
            for attribute, value_list in item.hyperparameters.items():
                pipeline_dict[item.name+'__' + attribute] = value_list
        self.pipeline_param_list = pipeline_dict
        return pipeline_dict

    def create_pipeline_elements_from_config(self, config):
        for key, all_params in config.items():
            self += PipelineElement(key, all_params)


class Hyperpipe(object):

    def __init__(self, pipe, specific_config, verbose=2, fit_params={}, error_score='raise'):

        self.params = specific_config
        self.pipe = pipe
        # print(self.params)

        # default
        self.return_train_score = True
        self.verbose = verbose
        self.fit_params = fit_params
        self.error_score = error_score

    def calculate_cv_loss(self, X, y, cv_iter):
        scores = []
        for train, test in cv_iter:
            fit_and_predict_score = _fit_and_score(clone(self.pipe), X, y, self.score,
                                                   train, test, self.verbose, self.params,
                                                   fit_params=self.fit_params,
                                                   return_train_score=self.return_train_score,
                                                   return_n_test_samples=True,
                                                   return_times=True, return_parameters=True,
                                                   error_score=self.error_score)
            scores.append(fit_and_predict_score)
        return scores

    def score(self, estimator, X_test, y_test):
        return estimator.score(X_test, y_test)

#
# T = TypeVar('T')


class PipelineElement(object):

    ELEMENT_DICTIONARY = {'pca': PCA,
                          'dnn': TFDNNClassifier}

    # def __new__(cls, name, position, hyperparameters, **kwargs):
    #     # print(cls)
    #     # print(*args)
    #     # print(**kwargs)
    #     desired_class = cls.ELEMENT_DICTIONARY[name]
    #     desired_class_instance = desired_class(**kwargs)
    #     desired_class_instance.name = name
    #     desired_class_instance.position = position
    #     return desired_class_instance

    def __init__(self, name, hyperparameters: dict, **kwargs):
        # Todo: check if adding position argument makes sense?
        # Todo: check if hyperparameters are members if the class
        # Todo: check if name is correctly mapped to class

        self.name = name
        self.hyperparameters = hyperparameters
        desired_class = self.ELEMENT_DICTIONARY[name]
        self.base_element = desired_class(**kwargs)
