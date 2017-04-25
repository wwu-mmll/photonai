
import numpy as np
from pprint import pprint
from itertools import product

from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import ParameterGrid
from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline
from HPOFramework.HPOptimizers import GridSearchOptimizer, RandomGridSearchOptimizer, TimeBoxedRandomGridSearchOptimizer
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.metrics import accuracy_score


class Hyperpipe(BaseEstimator):

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                            'random_grid_search': RandomGridSearchOptimizer,
                            'timeboxed_random_grid_search': TimeBoxedRandomGridSearchOptimizer}

    def __init__(self, name, cv_object: BaseCrossValidator, optimizer='grid_search', optimizer_params={},
                 local_search=True, groups=None,
                 config=None, X=None, y=None):

        self.name = name
        self.cv = cv_object
        self.cv_iter = None
        self.X = None
        self.y = None
        self.groups = groups

        self.config_history = []
        self.performance_history = []

        self.pipeline_elements = []
        self.pipeline_param_list = {}
        self.pipe = None
        self.optimum_pipe = None

        # Todo: this might be a case for sanity checking
        self.X = X
        self.y = y

        self._hyperparameters = []
        self._config_grid = []

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

        if isinstance(optimizer, str):
            # instantiate optimizer from string
            #  Todo: check if optimizer strategy is already implemented
            optimizer_class = self.OPTIMIZER_DICTIONARY[optimizer]
            optimizer_instance = optimizer_class(**optimizer_params)
            self.optimizer = optimizer_instance
            # we need an object for global search
            # so with a string it must be local search
            self.local_search = True
        else:
            # Todo: check if correct object
            self.optimizer = optimizer
            self.local_search = local_search

    def __iadd__(self, pipe_element):
        if isinstance(pipe_element, PipelineElement):
            self.pipeline_elements.append(pipe_element)
            # Todo: is repeated each time element is added....
            self.prepare_pipeline()
            return self
        else:
            # Todo: raise error
            raise TypeError("Element must be of type Pipeline Element")

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def add(self, pipe_element):
        self.__iadd__(pipe_element)

    def fit(self, data, targets, **fit_params):
        # prepare data ..

        # in case we don't want to use some data from outside the pipeline
        if self.X is None and self.y is None:
            self.X = data
            self.y = targets

        # Todo: use self.groups?
        self.cv_iter = list(self.cv.split(self.X, self.y))

        # optimize: iterate through configs and save results
        if self.local_search:
            # give the optimizer the chance to inform about elements
            self.optimizer.prepare(self.pipeline_elements)
            self.config_history = []
            self.performance_history = []

            for specific_config in self.optimizer.next_config:
                hp = TestPipeline(self.pipe, specific_config)
                print('******************************')
                print('optimizing of:', self.name)
                pprint(self.optimize_printing(specific_config))
                config_score = hp.calculate_cv_score(self.X, self.y, self.cv_iter)
                # 3. inform optimizer about performance
                self.optimizer.evaluate_recent_performance(specific_config, config_score)
                # inform user and log
                print('Training - %7.4f, Test - %7.4f' % (config_score[0], config_score[1]))
                print('--------------------------------------------------------------------')
                self.config_history.append(specific_config)
                self.performance_history.append(config_score)

            # afterwards find best result
            if len(self.performance_history) > 0:
                best_config_nr = np.argmax([t[1] for t in self.performance_history])
                best_config = self.config_history[best_config_nr]
                best_performance = self.performance_history[best_config_nr]
                # ... and create optimal pipeline
                # Todo: manage optimum pipe stuff
                self.optimum_pipe = self.pipe
                self.optimum_pipe.set_params(**best_config)

                # inform user
                print('--------------------------------------------------')
                print('Best config: ', best_config)
                print('Performance: Training - %7.4f, Test - %7.4f' % (best_performance[0], best_performance[1]))
                print('Number of tested configurations:', len(self.performance_history))
                print('--------------------------------------------------')
            # else:
                # raise Warning('Optimizer delivered no configurations to test. Is Pipeline empty?')

        else:
            self.pipe.fit(self.X, self.y, **fit_params)

        return self

    def predict(self, data):
        # Todo: if local_search = true then use optimized pipe here?
        if self.pipe is not None:
            return self.pipe.predict(data)
        else:
            return None

    def transform(self, data):
        if self.pipe is not None:
            return self.pipe.transform(data)
        else:
            return None

    def get_params(self, deep=True):
        if self.pipe is not None:
            return self.pipe.get_params(deep)
        else:
            return None

    def set_params(self, **params):
        if self.pipe is not None:
            self.pipe.set_params(**params)
        return self

    def prepare_pipeline(self):
        # prepare pipeline, hyperparams and config-grid
        self._config_grid = []
        self._hyperparameters = []
        pipeline_steps = []
        all_hyperparams = {}
        all_config_grids = []
        for item in self.pipeline_elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))
            all_hyperparams[item.name] = item.hyperparameters
            if item.config_grid:
                all_config_grids.append(item.config_grid)
        self._hyperparameters = all_hyperparams
        if len(all_config_grids) == 1:
            self._config_grid = all_config_grids[0]
        elif all_config_grids:
            # unpack list of dictionaries in one dictionary
            tmp_config_grid = list(product(*all_config_grids))
            for config_iterable in tmp_config_grid:
                base = dict(config_iterable[0])
                for i in range(1, len(config_iterable)):
                    base.update(config_iterable[i])
                self._config_grid.append(base)

        # build pipeline...
        self.pipe = Pipeline(pipeline_steps)

    def optimize_printing(self, config):
        prettified_config = []
        for el_key, el_value in config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in self.pipe.named_steps:
                new_pretty_key = self.name + '->' + name + '->'
                prettified_config.append(new_pretty_key +
                                          self.pipe.named_steps[name].prettify_config_output(rest, el_value))
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return prettified_config

    def prettify_config_output(self, config_name, config_value):
        return config_name + '=' + str(config_value)

    @property
    def config_grid(self):
        return self._config_grid

    def create_pipeline_elements_from_config(self, config):
        for key, all_params in config.items():
            self += PipelineElement(key, all_params)


    # @property
    # def optimum_pipe(self):
    #     return self.optimum_pipe


class TestPipeline(object):

    def __init__(self, pipe, specific_config, verbose=0, fit_params={}, error_score='raise'):

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
            # print('Data: \n')
            # print(np.reshape(X,(1, X.shape[0])))
        train_score_mean = np.mean([l[0] for l in scores])
        test_score_mean = np.mean([l[1] for l in scores])
        performance_tuple = (train_score_mean, test_score_mean)
        return performance_tuple

    def score(self, estimator, X_test, y_test):
        predicted = self.pipe.predict(X_test)
        return accuracy_score(y_test, predicted)
        # return estimator.score(X_test, y_test)

#
# T = TypeVar('T')


class PipelineElement(BaseEstimator):

    """
        Add any estimator or transform object from sklearn and associate unique name
        Add any own object that is compatible (implements fit and/or predict and/or fit_predict)
         and associate unique name
    """
    # from sklearn.decomposition import PCA
    # from sklearn.svm import SVC
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.preprocessing import StandardScaler
    # from TFLearnPipelineWrapper.WrapperModel import WrapperModel
    # from TFLearnPipelineWrapper.TFDNNClassifier import TFDNNClassifier
    # from TFLearnPipelineWrapper.KerasDNNWrapper import KerasDNNWrapper
    ELEMENT_DICTIONARY = {'pca': ('sklearn.decomposition', 'PCA'),
                          'svc': ('sklearn.svm', 'SVC'),
                          'logistic': ('sklearn.linear_model', 'LogisticRegression'),
                          'dnn': ('TFLearnPipelineWrapper.TFDNNClassifier', 'TFDNNClassifier'),
                          'kdnn': ('TFLearnPipelineWrapper.KerasDNNWrapper', 'KerasDNNWrapper'),
                          'standard_scaler': ('sklearn.preprocessing', 'StandardScaler'),
                          'wrapper_model': ('TFLearnPipelineWrapper.WrapperModel', 'WrapperModel')}

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
            try:
                desired_class_info = PipelineElement.ELEMENT_DICTIONARY[name]
                desired_class_home = desired_class_info[0]
                desired_class_name = desired_class_info[1]
                imported_module = __import__(desired_class_home, globals(), locals(), desired_class_name, 0)
                desired_class = getattr(imported_module, desired_class_name)
                base_element = desired_class(**kwargs)
                obj = PipelineElement(name, base_element, hyperparameters, set_disabled, disabled)
                return obj
            except AttributeError as ae:
                raise ValueError('Could not find according class:', PipelineElement.ELEMENT_DICTIONARY[name])
        else:
            raise NameError('Element not supported right now:', name)

    def __init__(self, name, base_element, hyperparameters: dict, set_disabled=False, disabled=False):
        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized --> sklearn: get_params.keys
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
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        # Todo: Make sure that set_disabled is not included when generating config_grid and stuff
        self._hyperparameters = value
        self.generate_sklearn_hyperparameters()
        self.generate_config_grid()
        if self.set_disabled:
            self._hyperparameters.update({'set_disabled': True})

    @property
    def config_grid(self):
        return self._config_grid

    @property
    def sklearn_hyperparams(self):
        return self._sklearn_hyperparams

    def generate_sklearn_hyperparameters(self):
        self._sklearn_hyperparams = {}
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
            if hasattr(self.base_element, 'transform'):
                return self.base_element.transform(data)
            elif hasattr(self.base_element, 'predict'):
                return self.base_element.predict(data)
            else:
                raise BaseException('transform-predict-mess')
        else:
            return data

    def score(self, X_test, y_test):
        return self.base_element.score(X_test, y_test)

    def prettify_config_output(self, config_name, config_value):
        return config_name + ':' + str(config_value)


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
        # Todo: return actual hyperparameters of all pipeline elements??
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_config_grid(self):
        hyperparameters = []
        for i, pipe_element in enumerate(self.pipeline_element_list):
            element_configurations = []
            for item in pipe_element.config_grid:
                element_configurations.append(item)
            self.pipeline_element_configurations.append(element_configurations)
            hyperparameters += [(i, nr) for nr in range(len(element_configurations))]
        self._config_grid = [{self._sklearn_curr_element: (i, nr)} for i, nr in hyperparameters]
        self._hyperparameters = {'current_element': hyperparameters}

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
        return obj

    def set_params(self, **kwargs):
        config_nr = None
        if self._sklearn_curr_element in kwargs:
            config_nr = kwargs[self._sklearn_curr_element]
        elif 'current_element' in kwargs:
            config_nr = kwargs['current_element']
        if config_nr is None or not isinstance(config_nr, tuple):
            raise ValueError('current_element must be of type Tuple')
        else:
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]
            # remove name
            unnamed_config = {}
            for config_key, config_value in config.items():
                key_split = config_key.split('__')
                unnamed_config[''.join(key_split[1::])] = config_value
            self.base_element.set_params(**unnamed_config)

    def prettify_config_output(self, config_name, config_value):
        if isinstance(config_value, tuple):
            output = self.pipeline_element_configurations[config_value[0]][config_value[1]]
            if not output:
                return self.pipeline_element_list[config_value[0]].name
            else:
                return str(output)
        else:
            return super(PipelineSwitch, self).prettify_config_output(config_name, config_value)


class PipelineFusion(PipelineElement):

    def __init__(self, name, pipeline_fusion_elements):
        super(PipelineFusion, self).__init__(name, None, hyperparameters={}, set_disabled=False, disabled=False)

        self._hyperparameters = {}
        self._config_grid = []
        self.pipe_elements = {}

        all_config_grids = []
        for item in pipeline_fusion_elements:
            self.pipe_elements[item.name] = item
            self._hyperparameters[item.name] = item.hyperparameters

            # for each configuration
            if not item.local_search:
                tmp_config_grid = []
                for config in item.config_grid:
                    # # for each configuration item:
                    # # if config is no dictionary -> unpack it
                    # if not isinstance(config, dict):
                    #     tmp_dict = {}
                    #     for c_item in config:
                    #         tmp_dict.update(c_item)
                    # else:
                    tmp_dict = dict(config)
                    tmp_config = dict(config)
                    for key, element in tmp_config.items():
                        # update name to be referable to pipeline
                        tmp_dict[self.name+'__'+item.name+'__'+key] = tmp_dict.pop(key)
                    tmp_config_grid.append(tmp_dict)
                all_config_grids.append(tmp_config_grid)
        if all_config_grids:
            self._config_grid = list(product(*all_config_grids))
            self._config_grid = [{**i[0], **i[1]} for i in self.config_grid]
        # for tmp_item in self._config_grid:
        #     tmp_2 = 1
        # tmp_i = 1

    @property
    def config_grid(self):
        return self._config_grid

    def get_params(self, deep=True):
        all_params = {}
        for name, element in self.pipe_elements.items():
            all_params[name] = element.get_params(deep)
        return all_params

    def set_params(self, **kwargs):
        # Todo: disable fusion element?
        spread_params_dict = {}
        for k, val in kwargs.items():
            splitted_k = k.split('__')
            item_name = splitted_k[0]
            if item_name not in spread_params_dict:
                spread_params_dict[item_name] = {}
            dict_entry = {'__'.join(splitted_k[1::]): val}
            spread_params_dict[item_name].update(dict_entry)

        for name, params in spread_params_dict.items():
            if name in self.pipe_elements:
                self.pipe_elements[name].set_params(**params)
            else:
                raise NameError('Could not find element ', name)
        return self

    def fit(self, data, targets):
        for name, element in self.pipe_elements.items():
            # Todo: parallellize fitting
            element.fit(data, targets)
        return self

    def predict(self, data):
        # Todo: strategy for concatenating data from different pipes
        predicted_data = None
        for name, element in self.pipe_elements.items():
            element_transform = element.predict(data)
            predicted_data = PipelineFusion.stack_data(predicted_data, element_transform)
        return predicted_data

    def transform(self, data):
        transformed_data = None
        for name, element in self.pipe_elements.items():
            # if it is a stacked pipeline then we want predict
            # element_transform = element.transform(data)
            element_transform = element.predict(data)
            transformed_data = PipelineFusion.stack_data(transformed_data, element_transform)
        return transformed_data

    @classmethod
    def stack_data(cls, a, b):
        if a is None:
            a = b
        else:
            # Todo: check for right dimensions!
            if a.ndim == 1 and b.ndim == 1:
                a = np.column_stack((a, b))
            else:
                a = np.hstack((a, b))
        return a

    def score(self, X_test, y_test):
        # Todo: invent strategy for this ?
        predicted = np.mean(self.predict(X_test))
        return accuracy_score(y_test, predicted)




