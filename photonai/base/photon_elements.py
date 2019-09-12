import importlib
import importlib.util
import inspect

import numpy as np


from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import ParameterGrid

from photonai.base.registry.element_dictionary import ElementDictionary
from photonai.base.helper import PhotonDataHelper
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.photonlogger import Logger
from photonai.optimization.config_grid import create_global_config_grid, create_global_config_dict


class PhotonNative:
    """only for checking if code is meeting requirements"""
    pass


class PipelineElement(BaseEstimator):
    """
    Photon wrapper class for any transformer or predictor element in the pipeline.

    1. Saves the hyperparameters that are to be tested and creates a grid of all hyperparameter configurations
    2. Enables fast and rapid instantiation of pipeline elements per string identifier,
         e.g 'svc' creates an sklearn.svm.SVC object.
    3. Attaches a "disable" switch to every element in the pipeline in order to test a complete disable


    Parameters
    ----------
    * `name` [str]:
       A string literal encoding the class to be instantiated
    * `hyperparameters` [dict]:
       Which values/value range should be tested for the hyperparameter.
       In form of "Hyperparameter_name: [array of parameter values to be tested]"
    * `test_disabled` [bool]:
        If the hyperparameter search should evaluate a complete disabling of the element
    * `disabled` [bool]:
        If true, the element is currently disabled and does nothing except return the data it received
    * `kwargs` [dict]:
        Any parameters that should be passed to the object to be instantiated, default parameters

    """
    # Registering Pipeline Elements
    ELEMENT_DICTIONARY = ElementDictionary.get_package_info()

    def __init__(self, name, hyperparameters: dict = None, test_disabled: bool = False,
                 disabled: bool = False, base_element=None, batch_size=0, **kwargs):
        """
        Takes a string literal and transforms it into an object of the associated class (see PhotonCore.JSON)

        Returns
        -------
        instantiated class object
        """
        if hyperparameters is None:
            hyperparameters = {}

        if base_element is None:
            if name in PipelineElement.ELEMENT_DICTIONARY:
                try:
                    desired_class_info = PipelineElement.ELEMENT_DICTIONARY[name]
                    desired_class_home = desired_class_info[0]
                    desired_class_name = desired_class_info[1]
                    imported_module = importlib.import_module(desired_class_home)
                    desired_class = getattr(imported_module, desired_class_name)
                    self.base_element = desired_class(**kwargs)
                except AttributeError as ae:
                    Logger().error('ValueError: Could not find according class:'
                                   + str(PipelineElement.ELEMENT_DICTIONARY[name]))
                    raise ValueError('Could not find according class:', PipelineElement.ELEMENT_DICTIONARY[name])
            else:
                Logger().error('Element not supported right now:' + name)
                raise NameError('Element not supported right now:', name)
        else:
            self.base_element = base_element

        self.is_transformer = hasattr(self.base_element, "transform")
        self.is_estimator = hasattr(self.base_element, "predict")

        self.kwargs = kwargs
        self.current_config = None
        self.batch_size = batch_size

        # Todo: write method that returns any hyperparameter that could be optimized --> sklearn: get_params.keys
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.test_disabled = test_disabled
        self._sklearn_disabled = self.name + '__disabled'
        self._hyperparameters = hyperparameters

        # check if hyperparameters are members of the class
        if self.is_transformer or self.is_estimator:
            self._check_hyperparameters(BaseEstimator)

        # self.initalize_hyperparameters = hyperparameters
        # check if hyperparameters are already in sklearn style
        if len(hyperparameters) > 0:
            key_0 = next(iter(hyperparameters))
            if self.name not in key_0:
                self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = hyperparameters
        self.disabled = disabled

        # check if self.base element needs y for fitting and transforming
        if hasattr(self.base_element, 'needs_y'):
            self.needs_y = self.base_element.needs_y
        else:
            self.needs_y = False
        # or if it maybe needs covariates for fitting and transforming
        if hasattr(self.base_element, 'needs_covariates'):
            self.needs_covariates = self.base_element.needs_covariates
        else:
            self.needs_covariates = False

    @property
    def _estimator_type(self):
        if hasattr(self.base_element, '_estimator_type'):
            est_type = getattr(self.base_element, '_estimator_type')
            if est_type is not 'classifier' and est_type is not 'regressor':
                raise NotImplementedError("Currently, we only support type classifier or regressor. Is {}.".format(est_type))
            if not hasattr(self.base_element, 'predict'):
                raise NotImplementedError("Estimator does not implement predict() method.")
            return est_type
        else:
            if hasattr(self.base_element, 'predict'):
                raise NotImplementedError("Element has predict() method but does not specify whether it is a regressor "
                                          "or classifier. Remember to inherit from ClassifierMixin or RegressorMixin.")
            else:
                return None

    def _check_hyperparameters(self, BaseEstimator):
        # check if hyperparameters are members of the class
        not_supported_hyperparameters = list(
            set([key.split("__")[-1] for key in self._hyperparameters.keys() if key.split("__")[-1] != "disabled"]) -
            set(BaseEstimator.get_params(self.base_element).keys()))
        if not_supported_hyperparameters:
            error_message = 'ValueError: Set of hyperparameters are not valid, check hyperparameters:' + \
                            str(not_supported_hyperparameters)
            Logger().error(error_message)
            raise ValueError(error_message)

    def copy_me(self):
        if self.name in self.ELEMENT_DICTIONARY:
            copy = PipelineElement(self.name, self.hyperparameters, **self.kwargs)
        else:
            # handle custom elements
            copy = PipelineElement.create(self.name, self.base_element, hyperparameters=self.hyperparameters, **self.kwargs)
        if self.current_config is not None:
            copy.set_params(**self.current_config)
        return copy

    @classmethod
    def create(cls, name, base_element, hyperparameters: dict, test_disabled=False, disabled=False, **kwargs):
        """
        Takes an instantiated object and encapsulates it into the PHOTON structure,
        add the disabled function and attaches information about the hyperparameters that should be tested
        """
        if isinstance(base_element, type):
            raise ValueError("Base element should be an instance but is a class.")
        return PipelineElement(name, hyperparameters, test_disabled, disabled, base_element=base_element, **kwargs)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: dict):
        self.generate_sklearn_hyperparameters(value)

    @property
    def feature_importances_(self):
        if hasattr(self.base_element, 'feature_importances_'):
            return self.base_element.feature_importances_

    @property
    def coef_(self):
        if hasattr(self.base_element, 'coef_'):
            return self.base_element.coef_

    def generate_config_grid(self):
        config_dict = create_global_config_dict([self])
        if len(config_dict) > 0:
            if self.test_disabled:
                config_dict.pop(self._sklearn_disabled)
            config_list = list(ParameterGrid(config_dict))
            if self.test_disabled:
                for item in config_list:
                    item[self._sklearn_disabled] = False
                config_list.append({self._sklearn_disabled: True})
                if len(config_list) < 2:
                    config_list.append({self._sklearn_disabled: False})

            return config_list
        else:
            return []

    def generate_sklearn_hyperparameters(self, value: dict):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        """
        self._hyperparameters = {}
        for attribute, value_list in value.items():
            self._hyperparameters[self.name + '__' + attribute] = value_list
        if self.test_disabled:
            self._hyperparameters[self._sklearn_disabled] = [False, True]

    def get_params(self, deep: bool=True):
        """
        Forwards the get_params request to the wrapped base element
        """
        return self.base_element.get_params(deep)

    def set_params(self, **kwargs):
        """
        Forwards the set_params request to the wrapped base element
        Takes care of the disabled parameter which is additionally attached by the PHOTON wrapper
        """
        # this is an ugly hack to approximate the right settings when copying the element
        self.current_config = kwargs
        # element disable is a construct used for this container only
        if self._sklearn_disabled in kwargs:
            self.disabled = kwargs[self._sklearn_disabled]
            del kwargs[self._sklearn_disabled]
        elif 'disabled' in kwargs:
            self.disabled = kwargs['disabled']
            del kwargs['disabled']
        self.base_element.set_params(**kwargs)
        return self

    def fit(self, X, y=None, **kwargs):
        """
        Calls the fit function of the base element

        Returns
        ------
        self
        """
        if not self.disabled:
            obj = self.base_element
            arg_list = inspect.signature(obj.fit)
            if len(arg_list.parameters) > 2:
                vals = arg_list.parameters.values()
                kwargs_param = list(vals)[-1]
                if kwargs_param.kind == kwargs_param.VAR_KEYWORD:
                    obj.fit(X, y, **kwargs)
                    return self
            obj.fit(X, y)
        return self

    def __batch_predict(self, delegate, X, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            Logger().warn("Cannot do batching on a single entity.")
            return delegate(X, **kwargs)

            # initialize return values
        processed_y = None
        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1
            Logger().debug(self.name + " is predicting batch nr " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(X, None, kwargs, start, stop)

            # predict
            y_pred = delegate(X_batched, **kwargs_dict_batched)
            processed_y = PhotonDataHelper.stack_results(y_pred, processed_y)

        return processed_y

    def __predict(self, X, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, 'predict'):
                # Todo: check if element has kwargs, and give it to them
                # todo: so this todo above was old, here are my changes:
                #return self.base_element.predict(X)
                return self.adjusted_predict_call(self.base_element.predict, X, **kwargs)
            else:
                Logger().error('BaseException. base Element should have function ' +
                               'predict.')
                raise BaseException('base Element should have function predict.')
        else:
            return X

    def predict(self, X, **kwargs):
        """
        Calls predict function on the base element.
        """
        if self.batch_size == 0:
            return self.__predict(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict, X, **kwargs)

    def predict_proba(self, X, **kwargs):
        if self.batch_size == 0:
            return self.__predict_proba(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict_proba(X, **kwargs))

    def __predict_proba(self, X, **kwargs):
        """
        Predict probabilities
        base element needs predict_proba() function, otherwise throw
        base exception.
        """
        if not self.disabled:
            if hasattr(self.base_element, 'predict_proba'):
                # todo: here, I used delegate call (same as below in predict within the transform call)
                #return self.base_element.predict_proba(X)
                return self.adjusted_predict_call(self.base_element.predict_proba, X, **kwargs)
            else:

                # todo: in case _final_estimator is a Branch, we do not know beforehand it the base elements will
                #  have a predict_proba -> if not, just return None (@Ramona, does this make sense?)
                #Logger().error('BaseException. base Element should have "predict_proba" function.')
                #raise BaseException('base Element should have predict_proba function.')
                return None
        return X

    def __transform(self, X, y=None, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, 'transform'):
                return self.adjusted_delegate_call(self.base_element.transform, X, y, **kwargs)
            elif hasattr(self.base_element, 'predict'):
                return self.predict(X, **kwargs), y, kwargs
            else:
                Logger().error('BaseException: transform-predict-mess')
                raise BaseException('transform-predict-mess')
        else:
            return X, y, kwargs

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on the base element.

        IN CASE THERE IS NO TRANSFORM METHOD, CALLS PREDICT.
        This is used if we are using an estimator as a preprocessing step.
        """
        if self.batch_size == 0:
            return self.__transform(X, y, **kwargs)
        else:
            return self.__batch_transform(X, y, **kwargs)

    def inverse_transform(self, X, y=None, **kwargs):
        if hasattr(self.base_element, 'inverse_transform'):
            # todo: check this
            X, y, kwargs = self.adjusted_delegate_call(self.base_element.inverse_transform, X, y, **kwargs)
        return X, y, kwargs

    def __batch_transform(self, X, y=None, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            Logger().warn("Cannot do batching on a single entity.")
            return self.__transform(X, y, **kwargs)

            # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1
            Logger().debug(self.name + " is transforming batch nr " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(X, y, kwargs, start, stop)

            # call transform
            X_new, y_new, kwargs_new = self.adjusted_delegate_call(self.base_element.transform, X_batched, y_batched,
                                                                   **kwargs_dict_batched)

            # stack results
            processed_X, processed_y, processed_kwargs = PhotonDataHelper.join_data(processed_X, X_new, processed_y,
                                                                                    y_new,
                                                                                    processed_kwargs, kwargs_new)

        return processed_X, processed_y, processed_kwargs

    def adjusted_delegate_call(self, delegate, X, y, **kwargs):
        # Case| transforms X | needs_y | needs_covariates
        # -------------------------------------------------------
        #   1         yes        no           no     = transform(X) -> returns Xt

        # todo: case does not exist any longer

        #   2         yes        yes          no     = transform(X, y) -> returns Xt, yt

        #   3         yes        yes          yes    = transform(X, y, kwargs) -> returns Xt, yt, kwargst
        #   4         yes        no           yes    = transform(X, kwargs) -> returns Xt, kwargst
        #   5         no      yes or no      yes or no      = NOT ALLOWED

        # todo: we don't need to check for Switch, Stack or Branch since those classes define
        # needs_y and needs_covariates in their __init__()
        if self.needs_y:
            # if we dont have any target vector, we are in "predict"-mode although we are currently transforming
            # in this case, we want to skip the transformation and pass X, None and kwargs onwards
            # so basically, we skip all training_only elements
            # todo: I think, there's no way around this if we want to pass y and kwargs down to children of Switch and Branch
            if isinstance(self, (Switch, Branch, Preprocessing)):
                X, y, kwargs = delegate(X, y, **kwargs)
            else:
                if y is not None:
                    # todo: in case a method needs y, we should also always pass kwargs
                    #  i.e. if we change the number of samples, we also need to apply that change to all kwargs
                    # todo: talk to Ramona! Maybe we actually DO need this case
                    if self.needs_covariates:
                        X, y, kwargs = delegate(X, y, **kwargs)
                    else:
                        X, y = delegate(X, y)
        elif self.needs_covariates:
            X, kwargs = delegate(X, **kwargs)

        else:
            X = delegate(X)

        return X, y, kwargs

    def adjusted_predict_call(self, delegate, X, **kwargs):
        if self.needs_covariates:
            return delegate(X, **kwargs)
        else:
            return delegate(X)

    def score(self, X_test, y_test):
        """
        Calls the score function on the base element:
        Returns a goodness of fit measure or a likelihood of unseen data:
        """
        return self.base_element.score(X_test, y_test)

    def prettify_config_output(self, config_name: str, config_value, return_dict: bool = False):
        """Make hyperparameter combinations human readable """
        if config_name == "disabled" and config_value is False:
            if return_dict:
                return {'disabled': False}
            else:
                return "disabled = False"
        else:
            if return_dict:
                return {config_name: config_value}
            else:
                return config_name + '=' + str(config_value)


class Branch(PipelineElement):
    """
     A substream of pipeline elements that is encapsulated e.g. for parallelization

     Parameters
     ----------
        * `name` [str]:
            Name of the encapsulated item and/or summary of the encapsulated element`s functions

        """

    def __init__(self, name, elements=None):

        super().__init__(name, {}, test_disabled=False, disabled=False, base_element=True)

        # in case any of the children needs y or covariates we need to request them
        self.needs_y = True
        self.needs_covariates = True
        self.elements = []
        self.has_hyperparameters = True
        self.skip_caching = True

        # needed for caching on individual level
        self.fix_fold_id = False
        self.do_not_delete_cache_folder = False
        
        # add elements
        if elements:
            for element in elements:
                self.add(element)

    def fit(self, X, y=None, **kwargs):
        return super().fit(X, y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        return super().transform(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)

    def __iadd__(self, pipe_element):
        """
        Add an element to the sub pipeline
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement or Hyperpipe]:
            The object to add, being either a transformer or an estimator.

        """
        self.elements.append(pipe_element)
        self._prepare_pipeline()
        return self

    def add(self, pipe_element):
        """
           Add an element to the sub pipeline
           Returns self

           Parameters
           ----------
           * `pipe_element` [PipelineElement or Hyperpipe]:
               The object to add, being either a transformer or an estimator.

           """
        self.__iadd__(pipe_element)

    def _prepare_pipeline(self):
        """ Generates sklearn pipeline with all underlying elements """
        pipeline_steps = []

        for item in self.elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))
            if hasattr(item, 'hyperparameters'):
                self._hyperparameters[item.name] = item.hyperparameters

        if self.has_hyperparameters:
            self.generate_sklearn_hyperparameters()
        new_pipe = PhotonPipeline(pipeline_steps)
        new_pipe._fix_fold_id = self.fix_fold_id
        new_pipe._do_not_delete_cache_folder = self.do_not_delete_cache_folder
        self.base_element = new_pipe

    def copy_me(self):
        new_copy_of_me = self.__class__(self.name)
        for item in self.elements:
            if hasattr(item, 'copy_me'):
                copy_item = item.copy_me()
            else:
                copy_item = deepcopy(item)
            new_copy_of_me += copy_item
        if self.current_config is not None:
            new_copy_of_me.set_params(**self.current_config)
        return new_copy_of_me

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        return

    @property
    def _estimator_type(self):
        return getattr(self.elements[-1], '_estimator_type')

    def generate_config_grid(self):
        if self.has_hyperparameters:
            tmp_grid = create_global_config_grid(self.elements, self.name)
            return tmp_grid
        else:
            return []

    def generate_sklearn_hyperparameters(self):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        """
        self._hyperparameters = {}
        for element in self.elements:
            for attribute, value_list in element.hyperparameters.items():
                self._hyperparameters[self.name + '__' + attribute] = value_list

    def _check_hyper(self,BaseEstimator):
        pass


class Preprocessing(Branch):
    """
        If a preprocessing pipe is added to a PHOTON Hyperpipe, all transformers are applied to the data ONCE
        BEFORE cross validation starts in order to prepare the data.
        Every added element should be a transformer PipelineElement.
    """

    def __init__(self):
        super().__init__('Preprocessing')
        self.has_hyperparameters = False
        self.needs_y = True
        self.needs_covariates = True
        self.name = 'Preprocessing'

    def __iadd__(self, pipe_element):
        """
        Add an element to the sub pipeline
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement]:
            The transformer object to add.

        """
        if hasattr(pipe_element, "transform"):
            if len(pipe_element.hyperparameters) > 0:
                raise ValueError("A preprocessing transformer must not have any hyperparameter "
                                 "because it is not part of the optimization and cross validation procedure")
            self.elements.append(pipe_element)
            self._prepare_pipeline()
        else:
            raise ValueError("Pipeline Element must have transform function")
        return self

    def predict(self, data, **kwargs):
        raise Warning("There is no predict function of the preprocessing pipe, it is a transformer only.")
        pass


class Stack(PipelineElement):
    """
    Creates a vertical stacking/parallelization of pipeline items.

    The object acts as single pipeline element and encapsulates several vertically stacked other pipeline elements, each
    child receiving the same input data. The data is iteratively distributed to all children, the results are collected
    and horizontally concatenated.

    """
    def __init__(self, name: str, elements=None):
        """
        Creates a new Stack element.
        Collects all possible hyperparameter combinations of the children

        Parameters
        ----------
        * `name` [str]:
            Give the pipeline element a name
        * `elements` [list, optional]:
            List of pipeline elements that should run in parallel
        * `voting` [bool]:
            If true, the predictions of the encapsulated pipeline elements are joined to a single prediction
        """
        super(Stack, self).__init__(name, hyperparameters={}, test_disabled=False, disabled=False,
                                    base_element=True)

        self._hyperparameters = {}
        self.elements = list()
        if elements is not None:
            for item_to_stack in elements:
                self.__iadd__(item_to_stack)

        # todo: Stack should not be allowed to change y, only covariates
        self.needs_y = False
        self.needs_covariates = True

    def __iadd__(self, item):
        """
        Adds a new element to the stack.
        Generates sklearn hyperparameter names in order to set the item's hyperparameters in the optimization process.

        * `item` [PipelineElement or Branch or Hyperpipe]:
            The Element that should be stacked and will run in a vertical parallelization in the original pipe.
        """
        self.check_if_needs_y(item)
        self.elements.append(item)

        # for each configuration
        tmp_dict = dict(item.hyperparameters)
        for key, element in tmp_dict.items():
            self._hyperparameters[self.name + '__' + key] = tmp_dict[key]

        return self

    def check_if_needs_y(self, item):
        if isinstance(item, (Branch, Stack, Switch)):
            for child_item in item.elements:
                self.check_if_needs_y(child_item)
        elif isinstance(item, PipelineElement):
            if item.needs_y:
                raise NotImplementedError("Elements in Stack must not transform y because the number of samples in every "
                                 "element of the stack might differ. Then, it will not be possible to concatenate those "
                                 "data and target matrices. Please use the transformer that is using y before or after "
                                 "the stack.")

    def add(self, item):
        self.__iadd__(item)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        pass

    def generate_config_grid(self):
        tmp_grid = create_global_config_grid(self.elements, self.name)
        return tmp_grid

    def get_params(self, deep=True):
        all_params = {}
        for element in self.elements:
            all_params[element.name] = element.get_params(deep)
        return all_params

    def set_params(self, **kwargs):
        """
        Find the particular child and distribute the params to it
        """
        spread_params_dict = {}
        for k, val in kwargs.items():
            splitted_k = k.split('__')
            item_name = splitted_k[0]
            if item_name not in spread_params_dict:
                spread_params_dict[item_name] = {}
            dict_entry = {'__'.join(splitted_k[1::]): val}
            spread_params_dict[item_name].update(dict_entry)

        for name, params in spread_params_dict.items():
            missing_element = (name, params)
            for element in self.elements:
                if element.name == name:
                    element.set_params(**params)
                    missing_element = None
            if missing_element:
                raise ValueError("Couldn't set hyperparameter for element {} -> {}".format(missing_element[0],
                                                                                           missing_element[1]))
        return self

    def fit(self, X, y=None, **kwargs):
        """
        Calls fit iteratively on every child
        """
        for element in self.elements:
            # Todo: parallellize fitting
            element.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Iteratively calls predict on every child.
        """
        # Todo: strategy for concatenating data from different pipes
        # todo: parallelize prediction
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict(X, **kwargs)
            predicted_data = Stack.stack_data(predicted_data, element_transform)
        if self.voting:
            if hasattr(predicted_data, 'shape'):
                if len(predicted_data.shape) > 1:
                    predicted_data = np.mean(predicted_data, axis=1).astype(int)
        return predicted_data, kwargs

    def predict_proba(self, X, y=None, **kwargs):
        """
        Predict probabilities for every pipe element and
        stack them together. Alternatively, do voting instead.
        """
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict_proba(X)
            predicted_data = Stack.stack_data(predicted_data, element_transform)
        if self.voting:
            if hasattr(predicted_data, 'shape'):
                if len(predicted_data.shape) > 1:
                    predicted_data = np.mean(predicted_data, axis=1).astype(int)
        return predicted_data

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on every child.

        If the encapsulated child is a hyperpipe, also calls predict on the last element in the pipeline.
        """
        transformed_data = np.array([])
        for element in self.elements:
            # if it is a hyperpipe with a final estimator, we want to use predict:
                element_transform, _, _ = element.transform(X, y, **kwargs)
                transformed_data = Stack.stack_data(transformed_data, element_transform)

        return transformed_data, y, kwargs

    def copy_me(self):
        ps = Stack(self.name)
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps.base_element = self.base_element
        return ps

    @classmethod
    def stack_data(cls, a, b):
        """
        Helper method to horizontally join the outcome of each child

        Parameters
        ----------
        * `a` [ndarray]:
            The existing matrix
        * `b` [ndarray]:
            The matrix that is to be attached horizontally

        Returns
        -------
        New matrix, that is a and b horizontally joined

        """
        if a is None or (isinstance(a, np.ndarray) and a.size == 0):
            a = b
        else:
            # Todo: check for right dimensions!
            if a.ndim == 1 and b.ndim == 1:
                a = np.column_stack((a, b))
            else:
                if b.ndim == 1:
                    b = np.reshape(b, (b.shape[0], 1))
                # a = np.concatenate((a, b), 1)
                a = np.concatenate((a, b), axis=1)
        return a

    def inverse_transform(self, X, y, **kwargs):
        raise NotImplementedError("Inverse Transform is not yet implemented for a Stacking Element in PHOTON")

    @property
    def _estimator_type(self):
        return None

    def _check_hyper(self,BaseEstimator):
        pass


class Switch(PipelineElement):
    """
    This class encapsulates several pipeline elements that belong at the same step of the pipeline,
    competing for being the best choice.

    If for example you want to find out if preprocessing A or preprocessing B is better at this position in the pipe.
    Or you want to test if a tree outperforms the good old SVM.

    ATTENTION: This class is a construct that may be convenient but is not suitable for any complex optimizations.
    Currently it only works for grid_search and the derived optimization strategies.
    USE THIS ONLY FOR RAPID PROTOTYPING AND PRELIMINARY RESULTS

    The class acts as if it is a single entity. Tt joins the hyperparamater combinations of each encapsulated element to
    a single, big combination grid. Each hyperparameter combination from that grid gets a number. Then the Switch
    object publishes the numbers to be chosen as the object's hyperparameter. When a new number is chosen from the
    optimizer, it internally activates the belonging element and sets the element's parameter to the hyperparameter
    combination. In that way, each of the elements is tested in all its configurations at the same position in the
    pipeline. From the outside, the process and the optimizer only sees one parameter of the Switch, that is
    the an integer indicating which item of the hyperparameter combination grid is currently active.

    """

    def __init__(self, name: str, elements: list = None):
        """
        Creates a new Switch object and generated the hyperparameter combination grid

        Parameters
        ----------
        * `name` [str]:
            How the element is called in the pipeline
        * `elements` [list, optional]:
            The competing pipeline elements
        * `_estimator_type:
            Used for validation purposes, either classifier or regressor

        """
        self.name = name
        self.sklearn_name = self.name + "__current_element"
        self._hyperparameters = {}
        self._current_element = (1, 1)
        self.pipeline_element_configurations = []
        self.base_element = None
        self.disabled = False
        self.test_disabled = False
        self.batch_size = 0

        self.needs_y = True
        self.needs_covariates = True
        # we assume we test models against each other, but only guessing
        self.is_estimator = True
        self.is_transformer = True

        self.elements_dict = {}

        if elements:
            self.elements = elements
            self.generate_private_config_grid()
            for p_element in elements:
                self.elements_dict[p_element.name] = p_element
        else:
            self.elements = []

    def __iadd__(self, pipeline_element):
        """
        Add a new estimator or transformer object to the switch container. All items change positions during testing.

        Parameters
        ----------
        * `pipeline_element` [PipelineElement]:
            Item that should be tested against other competing elements at that position in the pipeline.
        """
        self.elements.append(pipeline_element)
        if not pipeline_element.name in self.elements_dict:
            self.elements_dict[pipeline_element.name] = pipeline_element
        else:
            error_msg = "Already added a pipeline element with that name to the pipeline switch " + self.name
            Logger().error(error_msg)
            raise Exception(error_msg)
        self.generate_private_config_grid()
        return self

    def add(self, pipeline_element):
        """
        Add a new estimator or transformer object to the switch container. All items change positions during testing.

        Parameters
        ----------
        * `pipeline_element` [PipelineElement]:
            Item that should be tested against other competing elements at that position in the pipeline.
        """
        self.__iadd__(pipeline_element)

    @property
    def hyperparameters(self):
        # Todo: return actual hyperparameters of all pipeline elements??
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_private_config_grid(self):
        # reset
        self.pipeline_element_configurations = []

        # calculate anew
        hyperparameters = []
        # generate possible combinations for each item respectively - do not mix hyperparameters across items
        for i, pipe_element in enumerate(self.elements):
            # distinct_values_config = create_global_config([pipe_element])
            # add pipeline switch name in the config so that the hyperparameters can be set from other classes
            # pipeline switch will give the hyperparameters to the respective child
            # distinct_values_config_copy = {}
            # for config_key, config_value in distinct_values_config.items():
            #     distinct_values_config_copy[self.name + "__" + config_key] = config_value

            element_configurations = pipe_element.generate_config_grid()
            final_configuration_list = []
            if len(element_configurations) == 0:
                final_configuration_list.append({})
            # else:
            for dict_item in element_configurations:
                # copy_of_dict_item = {}
                # for key, value in dict_item.items():
                #     copy_of_dict_item[self.name + '__' + key] = value
                final_configuration_list.append(dict(dict_item))

            self.pipeline_element_configurations.append(final_configuration_list)
            hyperparameters += [(i, nr) for nr in range(len(final_configuration_list))]

        self._hyperparameters = {self.sklearn_name: hyperparameters}

    @property
    def current_element(self):
        return self._current_element

    @current_element.setter
    def current_element(self, value):
        self._current_element = value
        self.base_element = self.elements[self.current_element[0]]

    def get_params(self, deep: bool=True):
        if self.base_element:
            return self.base_element.get_params(deep)
        else:
            return {}

    def set_params(self, **kwargs):

        """
        The optimization process sees the amount of possible combinations and chooses one of them.
        Then this class activates the belonging element and prepared the element with the particular chosen configuration.

        """

        config_nr = None
        config = None

        # in case we are operating with grid search
        if self.sklearn_name in kwargs:
            config_nr = kwargs[self.sklearn_name]
        elif 'current_element' in kwargs:
            config_nr = kwargs['current_element']

        # in case we are operating with another optimizer
        if config_nr is None:

            # we need to identify the element to activate by checking for which element the optimizer gave params
            if kwargs is not None:
                config = kwargs
                # ugly hack because subscription is somehow not possible, we use the for loop but break
                for kwargs_key, kwargs_value in kwargs.items():
                    first_element_name = kwargs_key.split("__")[0]
                    self.base_element = self.elements_dict[first_element_name]
                    break
        else:
            if not isinstance(config_nr, (tuple, list)):
                Logger().error('ValueError: current_element must be of type Tuple')
                raise ValueError('current_element must be of type Tuple')

            # grid search hack
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]

        if config:
            # remove name
            unnamed_config = {}
            for config_key, config_value in config.items():
                key_split = config_key.split('__')
                unnamed_config['__'.join(key_split[1::])] = config_value
            self.base_element.set_params(**unnamed_config)
        return self

    def copy_me(self):

        ps = Switch(self.name)
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps.base_element = self.base_element
        return ps

    def prettify_config_output(self, config_name, config_value, return_dict=False):

        """
        Makes the sklearn configuration dictionary human readable

        Returns
        -------
        * `prettified_configuration_string` [str]:
            configuration as prettified string or configuration as dict with prettified keys
        """

        if isinstance(config_value, tuple):
            output = self.pipeline_element_configurations[config_value[0]][config_value[1]]
            if not output:
                if return_dict:
                    return {self.elements[config_value[0]].name:None}
                else:
                    return self.elements[config_value[0]].name
            else:
                if return_dict:
                    return output
                return str(output)
        else:
            return super(Switch, self).prettify_config_output(config_name, config_value)

    def predict_proba(self, X, **kwargs):
        """
        Predict probabilities
        base element needs predict_proba() function, otherwise throw
        base exception.
        """
        if not self.disabled:
            if hasattr(self.base_element.base_element, 'predict_proba'):
                return self.base_element.predict_proba(X)
            else:
                return None
        return X

    def _check_hyper(self,BaseEstimator):
        pass

    def inverse_transform(self, X, y=None, **kwargs):
        if hasattr(self.base_element, 'inverse_transform'):
            # todo: check this
            X, y, kwargs = self.adjusted_delegate_call(self.base_element.inverse_transform, X, y, **kwargs)
        return X, y, kwargs

    @property
    def _estimator_type(self):
        estimator_types = list()
        for element in self.elements:
            estimator_types.append(getattr(element, '_estimator_type'))

        unique_types = set(estimator_types)
        if len(unique_types) > 1:
            raise NotImplementedError("Switch should only contain elements of a single type (transformer, classifier, "
                                      "regressor). Found multiple types: {}".format(unique_types))
        elif len(unique_types) == 1:
            return list(unique_types)[0]
        else:
            return None


class DataFilter(BaseEstimator):
    """
    Helper Class to split the data e.g. for stacking.
    """
    def __init__(self, indices):
        self.name = 'DataFilter'
        self.hyperparameters = {}
        self.indices = indices
        self.needs_covariates = False
        self.needs_y = False

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Returns only part of the data, column-wise filtered by self.indices
        """
        return X[:, self.indices], y, kwargs

    def copy_me(self):
        return self.__class__(indices=self.indices)

    @property
    def _estimator_type(self):
        return None


class CallbackElement(PhotonNative):

    def __init__(self, name, delegate_function, method_to_monitor='transform'):

        self.needs_covariates = True
        self.needs_y = True
        self.name = name
        self.delegate_function = delegate_function
        self.method_to_monitor = method_to_monitor
        self.hyperparameters = {}
        self.is_transformer = True
        self.is_estimator = False

    def fit(self, X, y=None, **kwargs):
        if self.method_to_monitor == 'fit':
            self.delegate_function(X, y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        if self.method_to_monitor == 'transform':
            self.delegate_function(X, y, **kwargs)
        return X, y, kwargs

    def copy_me(self):
        return self.__class__(self.name, self.delegate_function, self.method_to_monitor)

    def inverse_transform(self, X, y=None, **kwargs):
        return X, y, kwargs

    @property
    def _estimator_type(self):
        return None

