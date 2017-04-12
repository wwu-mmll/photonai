
from typing import Generic, TypeVar

import numpy as np

from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import ParameterGrid
from sklearn.base import clone, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from HPOFramework.HPOptimizers import GridSearchOptimizer
from TFLearnPipelineWrapper.TFDNNClassifier import TFDNNClassifier
from TFLearnPipelineWrapper.KerasDNNWrapper import KerasDNNWrapper
from sklearn.model_selection._split import BaseCrossValidator


class HyperpipeManager(BaseEstimator):

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer}

    def __init__(self, cv_object: BaseCrossValidator, optimizer='grid_search', groups=None, config=None):

        self.cv = cv_object
        self.cv_iter = None
        self.X = None
        self.y = None
        self.groups = groups

        self.pipeline_elements = []
        self.pipeline_param_list = {}
        self.pipe = None
        self.optimum_pipe = None

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

        if isinstance(optimizer, str):
            # instantiate optimizer from string
            #  Todo: check if optimizer strategy is already implemented
            optimizer_class = self.OPTIMIZER_DICTIONARY[optimizer]
            optimizer_instance = optimizer_class()
            self.optimizer = optimizer_instance
        else:
            # Todo: check if correct object
            self.optimizer = optimizer

    def __iadd__(self, pipe_element):
        if isinstance(pipe_element, PipelineElement):
            self.pipeline_elements.append(pipe_element)
            return self
        else:
            # Todo: raise error
            raise TypeError("Element must be of type Pipeline Element")

    def add(self, pipe_element):
        self.__iadd__(pipe_element)

    def fit(self, data, targets):
        # prepare data ..
        self.X = data
        self.y = targets
        # Todo: use self.groups?
        self.cv_iter = list(self.cv.split(self.X, self.y))

        # 0. build pipeline...
        pipeline_steps = []
        for item in self.pipeline_elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))
        self.pipe = Pipeline(pipeline_steps)

        # and bring hyperparameters into sklearn pipeline syntax
        self.organize_parameters()
        self.optimizer.prepare(self.pipeline_elements)

        config_history = []
        performance_history = []

        # 2. iterate next_config and save results
        for specific_config in self.optimizer.next_config:
            hp = TestPipeline(self.pipe, specific_config)
            config_score = hp.calculate_cv_score(self.X, self.y, self.cv_iter)
            # 3. inform optimizer about performance
            self.optimizer.evaluate_recent_performance(specific_config, config_score)
            # inform user and log
            print(config_score)
            config_history.append(specific_config)
            performance_history.append(config_score)

        # 4. find best result
        best_config_nr = np.argmax([t[1] for t in performance_history])
        best_config = config_history[best_config_nr]
        best_performance = performance_history[best_config_nr]
        # ... and create optimal pipeline
        # Todo: manage optimum pipe stuff
        self.optimum_pipe = self.pipe
        self.optimum_pipe.set_params(**best_config)

        # inform user
        print('--------------------------------------------------')
        print('Best config: ', best_config)
        print('Performance: Training - %7.4f, Test - %7.4f' % (best_performance[0], best_performance[1]))
        print('--------------------------------------------------')

    def organize_parameters(self):
        pipeline_dict = dict()
        for item in self.pipeline_elements:
            pipeline_dict.update(item.sklearn_hyperparams)
        self.pipeline_param_list = pipeline_dict
        return pipeline_dict

    def create_pipeline_elements_from_config(self, config):
        for key, all_params in config.items():
            self += PipelineElement(key, all_params)

    # @property
    # def optimum_pipe(self):
    #     return self.optimum_pipe


class TestPipeline(object):

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
            # why clone? removed: clone(self.pipe),
            fit_and_predict_score = _fit_and_score(self.pipe, X, y, self.score,
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
    @classmethod
    def create(cls, name, hyperparameters: dict ={}, set_disabled=False, disabled=False, **kwargs):
        if name in PipelineElement.ELEMENT_DICTIONARY:
            desired_class = PipelineElement.ELEMENT_DICTIONARY[name]
            base_element = desired_class(**kwargs)
            obj = PipelineElement(name, base_element, hyperparameters, set_disabled, disabled)
            return obj
        else:
            raise NameError('Element not supported right now:', name)

    def __init__(self, name, base_element, hyperparameters: dict, set_disabled=False, disabled=False):
        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.base_element = base_element
        self.disabled = disabled
        self.set_disabled = set_disabled
        self._hyperparameters = hyperparameters
        self._sklearn_hyperparams = {}
        self._sklearn_disabled = self.name + '__disabled'
        self._config_grid = []
        self.hyperparameters = self._hyperparameters


    @property
    def hyperparameters(self):
        # if self.disable:
        #     return {}
        # else:
        #     return self._hyperparameters
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        self._hyperparameters = value
        self.generate_sklearn_hyperparameters()
        self.generate_config_grid()

    @property
    def config_grid(self):
        return self._config_grid

    @property
    def sklearn_hyperparams(self):
        return self._sklearn_hyperparams

    def generate_sklearn_hyperparameters(self):
        for attribute, value_list in self._hyperparameters.items():
            self._sklearn_hyperparams[self.name + '__' + attribute] = value_list

    def generate_config_grid(self):
        for item in ParameterGrid(self.sklearn_hyperparams):
            if self.set_disabled:
                item[self._sklearn_disabled] = False
            self._config_grid.append(item)
        if self.set_disabled:
            self._config_grid.append({self._sklearn_disabled: True})

    def get_params(self, deep=True):
        return self.base_element.get_params(deep)

    def set_params(self, **kwargs):
        # element disable is a construct used for this container only
        if self._sklearn_disabled in kwargs:
            self.disabled = kwargs[self._sklearn_disabled]
            del kwargs[self._sklearn_disabled]
        elif 'disabled' in kwargs:
            self.disabled = kwargs['disabled']
            del kwargs['disabled']
        self.base_element.set_params(**kwargs)
        return self

    def fit(self, data, targets):
        if not self.disabled:
            obj = self.base_element
            obj.fit(data, targets)
            # self.base_element.fit(data, targets)
        return self

    def predict(self, data):
        if not self.disabled:
            return self.base_element.predict(data)
        else:
            return data

    def transform(self, data):
        if not self.disabled:
            return self.base_element.transform(data)
        else:
            return data

    def score(self, X_test, y_test):
        return self.base_element.score(X_test, y_test)


class PipelineSwitch(PipelineElement):

    # @classmethod
    # def create(cls, pipeline_element_list):
    #     obj = PipelineSwitch()
    #     obj.pipeline_element_list = pipeline_element_list
    #     return obj

    def __init__(self, name, pipeline_element_list):
        self.name = name
        self._sklearn_curr_element = self.name + '__current_element'
        # Todo: disable switch?
        self.disabled = False
        self.set_disabled = False
        self._hyperparameters = {}
        self._sklearn_hyperparams = {}
        self.hyperparameters = self._hyperparameters
        self._config_grid = []
        self._current_element = (1, 1)
        self.pipeline_element_list = pipeline_element_list
        self.pipeline_element_configurations = []
        self.generate_config_grid()
        self.generate_sklearn_hyperparameters()


    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_config_grid(self):
        hyperparameters = []
        for i, pipe_element in enumerate(self.pipeline_element_list):
            element_configurations = []
            if len(pipe_element.hyperparameters) > 0:
                for item in ParameterGrid(pipe_element.hyperparameters):
                    element_configurations.append(item)
            self.pipeline_element_configurations.append(element_configurations)
            hyperparameters += [(i, nr) for nr in range(len(element_configurations))]
        self._config_grid = [{self._sklearn_curr_element: (i, nr)} for i, nr in hyperparameters]
        self._hyperparameters = {self._sklearn_curr_element: hyperparameters}

    @property
    def current_element(self):
        return self._current_element

    @property
    def config_grid(self):
        return self._config_grid

    @current_element.setter
    def current_element(self, value):
        self._current_element = value
        # pass the right config to the element
        # config = self.pipeline_element_configurations[value[0]][value[1]]
        # self.base_element.set_params(config)

    @property
    def base_element(self):
        obj = self.pipeline_element_list[self.current_element[0]]
        return obj.base_element

    def set_params(self, **kwargs):
        config_nr = None
        if self._sklearn_curr_element in kwargs:
            config_nr = kwargs[self._sklearn_curr_element]
        elif 'current_element' in kwargs:
            config_nr = kwargs['current_element']
        if config_nr is not None:
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]
            self.base_element.set_params(**config)



#
# class PipelineFusion(HyperpipeManager):
#
#     def __init__(self, cv_object: BaseCrossValidator, optimizer='grid_search', groups=None, config=None):
#         super.__init__(cv_object, optimizer, groups, config)
