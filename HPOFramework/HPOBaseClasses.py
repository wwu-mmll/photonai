
from typing import Generic, TypeVar

import numpy as np

from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from HPOFramework.HPOptimizers import GridSearchOptimizer
from TFLearnPipelineWrapper.TFDNNClassifier import TFDNNClassifier
from TFLearnPipelineWrapper.KerasDNNWrapper import KerasDNNWrapper


class HyperpipeManager(BaseEstimator):

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer}

    def __init__(self, data_container, groups=None, config=None):

        # self.param_dict = param_dict
        self.data_container = data_container
        self.pipeline_param_list = {}
        self.X = data_container.features.values
        self.y = np.ravel(data_container.targets.values)
        self.groups = groups

        # Todo: implement CV adjustment strategy
        self.cv = KFold(n_splits=3)
        self.cv_iter = list(self.cv.split(self.X, self.y, groups))

        self.pipeline_elements = []
        self.pipe = None
        self.optimum_pipe = None

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

    def __iadd__(self, pipe_element):
        if isinstance(pipe_element, PipelineElement):
            self.pipeline_elements.append(pipe_element)
            return self
        else:
            #Todo: raise error
            raise TypeError("Element must be of type Pipeline Element")

    def add(self, pipe_element):
        self.__iadd__(pipe_element)

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

        config_history = []
        performance_history = []

        # 2. iterate next_config and save results
        for specific_config in optimizer_instance.next_config:
            hp = Hyperpipe(self.pipe, specific_config)
            config_score = hp.calculate_cv_score(self.X, self.y, self.cv_iter)
            # 3. inform optimizer about performance
            optimizer_instance.evaluate_recent_performance(specific_config, config_score)
            # inform user and log
            print(config_score)
            config_history.append(specific_config)
            performance_history.append(config_score)

        # 4. find best result
        best_config_nr = np.argmax([t[1] for t in performance_history])
        best_config = config_history[best_config_nr]
        best_performance = performance_history[best_config_nr]
        # ... and create optimal pipeline
        self.optimum_pipe = clone(self.pipe)
        self.optimum_pipe.set_params(**best_config)

        # inform user
        print('--------------------------------------------------')
        print('Best config: ', best_config)
        print('Performance: Training - %7.4f, Test - %7.4f' % (best_performance[0], best_performance[1]))
        print('--------------------------------------------------')

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

    # @property
    # def optimum_pipe(self):
    #     return self.optimum_pipe


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

    def calculate_cv_score(self, X, y, cv_iter):
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
        train_score_mean = np.mean([l[0] for l in scores])
        test_score_mean = np.mean([l[1] for l in scores])
        performance_tuple = (train_score_mean, test_score_mean)
        return performance_tuple

    def score(self, estimator, X_test, y_test):
        return estimator.score(X_test, y_test)

#
# T = TypeVar('T')


class PipelineElement(object):

    """
        Add any estimator or transform object from sklearn and associate unique name
        Add any own object that is compatible (implements fit and/or predict and/or fit_predict)
         and associate unique name
    """
    ELEMENT_DICTIONARY = {'pca': PCA,
                          'svc': SVC,
                          'logistic': LogisticRegression,
                          'dnn': TFDNNClassifier,
                          'kdnn': KerasDNNWrapper}

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
        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.hyperparameters = hyperparameters
        if name in self.ELEMENT_DICTIONARY:
            desired_class = self.ELEMENT_DICTIONARY[name]
            self.base_element = desired_class(**kwargs)
        else:
            self.base_element = None
            raise NameError('Element not supported right now:', name)

