import importlib
import importlib.util
import inspect
from copy import deepcopy
import dask
from dask.distributed import Client
import numpy as np
import warnings
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import ParameterGrid
from typing import List, Union

from photonai.base.photon_pipeline import PhotonPipeline
from photonai.base.registry.registry import PhotonRegistry
from photonai.helper.helper import PhotonDataHelper
from photonai.optimization.config_grid import create_global_config_grid, create_global_config_dict
from photonai.photonlogger.logger import logger


class PhotonNative:
    """Only for checking if code is meeting requirements."""
    pass


class PipelineElement(BaseEstimator):
    """
    PHOTONAI wrapper class for any transformer or estimator in the pipeline.

    So called PHOTONAI PipelineElements can be added to the Hyperpipe,
    each of them being a data-processing method or a learning algorithm.
    By choosing, combining data-processing methods and algorithms,
    and arranging them with the PHOTONAI classes, simple and complex
    pipeline architectures can be designed rapidly.

    The PHOTONAI PipelineElement implements several helpful features:

    - Saves the hyperparameters that should be tested
        and creates a grid of all hyperparameter configurations.
    - Enables fast and rapid instantiation of pipeline
        elements per string identifier, e.g 'svc' creates
        an sklearn.svm.SVC object.
    - Attaches a "disable" switch to every element
        in the pipeline in order to test a complete disable.

    """
    def __init__(self, name: str, hyperparameters: dict = None, test_disabled: bool = False,
                 disabled: bool = False, base_element: BaseEstimator = None, batch_size: int = 0, **kwargs) -> None:
        """
        Takes a string literal and transforms it into an object
        of the associated class (see PhotonCore.JSON).

        Parameters:
            name:
                A string literal encoding the class to be instantiated.

            hyperparameters:
                Which values/value range should be tested for the
                hyperparameter.
                In form of Dict: parameter_name -> HyperparameterElement.

            test_disabled:
                If the hyperparameter search should evaluate a
                complete disabling of the element.

            disabled:
                If true, the element is currently disabled and
                does nothing except return the data it received.

            base_element:
                The underlying BaseEstimator. If not given the
                instantiation per string identifier takes place.

            batch_size:
                Size of the division on which is calculated separately.

            **kwargs:
                Any parameters that should be passed to the object
                to be instantiated, default parameters.

        """
        if hyperparameters is None:
            hyperparameters = {}

        if base_element is None:

            # Registering Pipeline Elements
            if len(PhotonRegistry.ELEMENT_DICTIONARY) == 0:
                registry = PhotonRegistry

            if name not in PhotonRegistry.ELEMENT_DICTIONARY:
                # try to reload
                PhotonRegistry.ELEMENT_DICTIONARY = PhotonRegistry().get_package_info()

            if name in PhotonRegistry.ELEMENT_DICTIONARY:
                try:
                    desired_class_info = PhotonRegistry.ELEMENT_DICTIONARY[name]
                    desired_class_home = desired_class_info[0]
                    desired_class_name = desired_class_info[1]
                    imported_module = importlib.import_module(desired_class_home)
                    desired_class = getattr(imported_module, desired_class_name)
                    self.base_element = desired_class(**kwargs)
                except AttributeError as ae:
                    logger.error('ValueError: Could not find according class:'
                                 + str(PhotonRegistry.ELEMENT_DICTIONARY[name]))
                    raise ValueError('Could not find according class:', PhotonRegistry.ELEMENT_DICTIONARY[name])
            else:
                # if even after reload the element does not appear, it is not supported
                logger.error('Element not supported right now:' + name)
                raise NameError('Element not supported right now:', name)
        else:
            self.base_element = base_element

        self.is_transformer = hasattr(self.base_element, "transform")
        self.reduce_dimension = False  # boolean - set on transform method
        self.is_estimator = hasattr(self.base_element, "predict")
        self._name = name
        self.initial_name = str(name)
        self.kwargs = kwargs
        self.current_config = None
        self.batch_size = batch_size
        self.test_disabled = test_disabled
        self.initial_hyperparameters = dict(hyperparameters)

        self._sklearn_disabled = self.name + '__disabled'
        self._hyperparameters = hyperparameters
        if len(hyperparameters) > 0:
            key_0 = next(iter(hyperparameters))
            if self.name not in key_0:
                self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = hyperparameters
        # self.initalize_hyperparameters = hyperparameters
        # check if hyperparameters are already in sklearn style

        # check if hyperparameters are members of the class
        if self.is_transformer or self.is_estimator:
            self._check_hyperparameters(BaseEstimator)

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

        self._random_state = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.generate_sklearn_hyperparameters(self.initial_hyperparameters)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: dict):
        self.generate_sklearn_hyperparameters(value)

    def _check_hyperparameters(self, BaseEstimator):
        # check if hyperparameters are members of the class
        not_supported_hyperparameters = list(
            set([key.split("__")[-1] for key in self._hyperparameters.keys() if key.split("__")[-1] != "disabled"]) -
            set(BaseEstimator.get_params(self.base_element).keys()))
        if not_supported_hyperparameters:
            error_message = 'ValueError: Set of hyperparameters are not valid, check hyperparameters:' + \
                            str(not_supported_hyperparameters)
            logger.error(error_message)
            raise ValueError(error_message)

    def generate_sklearn_hyperparameters(self, value: dict):
        """
        Generates a dictionary according to the sklearn convention of
        element_name__parameter_name: parameter_value.
        """
        self._hyperparameters = {}
        for attribute, value_list in value.items():
            self._hyperparameters[self.name + '__' + attribute] = value_list
        if self.test_disabled:
            self._hyperparameters[self._sklearn_disabled] = [False, True]

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
        if hasattr(self, 'elements'):
            for el in self.elements:
                if hasattr(el, 'random_state'):
                    el.random_state = self._random_state
        if hasattr(self, "base_element") and hasattr(self.base_element, "random_state"):
            self.base_element.random_state = random_state

    @property
    def _estimator_type(self):
        # estimator_type obligation for estimators, is ignored if a transformer is given
        # prevention of misuse through predict test (predict method available <=> Estimator).
        est_type = getattr(self.base_element, '_estimator_type', None)
        if est_type in [None, 'transformer']:
            if hasattr(self.base_element, 'predict'):
                raise NotImplementedError("Element has predict() method but does not specify whether it is a regressor"
                                          " or classifier. Remember to inherit from ClassifierMixin or RegressorMixin.")
            return None
        else:
            if est_type not in ['classifier', 'regressor']:
                raise NotImplementedError("Currently, we only support type classifier or regressor."
                                          " Is {}.".format(est_type))
            if not hasattr(self.base_element, 'predict'):
                raise NotImplementedError("Estimator does not implement predict() method.")
            return est_type

    # this is only here because everything inherits from PipelineElement.
    def __iadd__(self, pipe_element):
        """
        Add an element to the intern list of elements.

        Parameters:
            pipe_element (PipelineElement):
                The object to add, being either a transformer or an estimator.

        """
        PipelineElement.sanity_check_element_type_for_building_photon_pipes(pipe_element, type(self))

        # check if that exact instance has been added before
        already_added_objects = len([i for i in self.elements if i is pipe_element])
        if already_added_objects > 0:
            error_msg = "Cannot add the same instance twice to " + self.name + " - " + str(type(self))
            logger.error(error_msg)
            raise ValueError(error_msg)

        # check for doubled names:
        already_existing_element_with_that_name = len([i for i in self.elements if i.name == pipe_element.name])

        if already_existing_element_with_that_name > 0:
            error_msg = "Already added a pipeline element with the name " + pipe_element.name + " to " + self.name
            logger.warning(error_msg)
            warnings.warn(error_msg)

            # check for other items that have been renamed
            nr_of_existing_elements_with_that_name = len([i for i in self.elements if i.name.startswith(pipe_element.name)])
            new_name = pipe_element.name + str(nr_of_existing_elements_with_that_name + 1)
            while len([i for i in self.elements if i.name == new_name]) > 0:
                nr_of_existing_elements_with_that_name += 1
                new_name = pipe_element.name + str(nr_of_existing_elements_with_that_name + 1)
            msg = "Renaming " + pipe_element.name + " in " + self.name + " to " + new_name + " in " + self.name
            logger.warning(msg)
            warnings.warn(msg)
            pipe_element.name = new_name

        self.elements.append(pipe_element)
        return self

    def copy_me(self):
        if self.name in PhotonRegistry.ELEMENT_DICTIONARY:
            # we need initial name to refer to the class to be instantiated  (SVC) even though the name might be SVC2
            copy = PipelineElement(self.initial_name, {}, test_disabled=self.test_disabled,
                                   disabled=self.disabled, batch_size=self.batch_size, **self.kwargs)
            copy.initial_hyperparameters = self.initial_hyperparameters
            # in the setter of the name, we use initial hyperparameters to adjust the hyperparameters to the name
            copy.name = self.name
        else:
            if hasattr(self.base_element, 'copy_me'):
                new_base_element = self.base_element.copy_me()
            else:
                try:
                    new_base_element = deepcopy(self.base_element)
                except Exception as e:
                    error_msg = "Cannot copy custom element " + self.name + ". Please specify a copy_me() method " \
                                                                        "returning a copy of the object"
                    logger.error(error_msg)
                    raise e

            # handle custom elements
            copy = PipelineElement.create(self.name, new_base_element, hyperparameters=self.hyperparameters,
                                          test_disabled=self.test_disabled,
                                          disabled=self.disabled, batch_size=self.batch_size,
                                          **self.kwargs)
        if self.current_config is not None:
            copy.set_params(**self.current_config)
        copy._random_state = self._random_state
        return copy

    @classmethod
    def create(cls, name: str, base_element: BaseEstimator, hyperparameters: dict, test_disabled: bool = False,
               disabled: bool = False, **kwargs):
        """
        Takes an instantiated object and encapsulates it
        into the PHOTONAI structure.
        Add the disabled function and attaches
        information about the hyperparameters that should be tested.

        Parameters:
            name:
                A string literal encoding the class to be instantiated.

            base_element:
                The underlying transformer or estimator class.

            hyperparameters:
                Which values/value range should be tested for the
                hyperparameter.
                In form of Dict: parameter_name -> HyperparameterElement.

            test_disabled:
                If the hyperparameter search should evaluate a
                complete disabling of the element.

            disabled:
                If true, the element is currently disabled and
                does nothing except return the data it received.

            **kwargs:
                Any parameters that should be passed to the object
                to be instantiated, default parameters.

        Example:
            ``` python
            class RD(BaseEstimator, TransformerMixin):

                def fit(self, X, y, **kwargs):
                    pass

                def fit_transform(self, X, y=None, **fit_params):
                    return self.transform(X)

                def transform(self, X):
                    return X[:, :3]

            trans = PipelineElement.create('MyTransformer', base_element=RD(), hyperparameters={})
            ```

        """
        if isinstance(base_element, type):
            raise ValueError("Base element should be an instance but is a class.")
        return PipelineElement(name, hyperparameters, test_disabled, disabled, base_element=base_element, **kwargs)

    @property
    def feature_importances_(self):
        if hasattr(self.base_element, 'feature_importances_'):
            return self.base_element.feature_importances_.tolist()
        elif hasattr(self.base_element, 'coef_'):
            return self.base_element.coef_.tolist()

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

    def get_params(self, deep: bool = True):
        """
        Forwards the get_params request to the wrapped base element.
        """
        if hasattr(self.base_element, 'get_params'):
            params = self.base_element.get_params(deep)
            params["name"] = self.name
            return params
        else:
            return None

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

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Calls the fit function of the base element.

        Parameters:
            X:
                The array-like training and test data with shape=[N, D],
                where N is the number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to base_element.predict.

        Returns:
            Fitted self.

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
            msg = "Cannot do batching on a single entity."
            logger.warning(msg)
            warnings.warn(msg)
            return delegate(X, **kwargs)

            # initialize return values
        processed_y = None
        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1
            logger.debug(self.name + " is predicting batch " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(X, None, kwargs, start, stop)

            # predict
            y_pred = delegate(X_batched, **kwargs_dict_batched)
            processed_y = PhotonDataHelper.stack_data_vertically(processed_y, y_pred)

        return processed_y

    def __predict(self, X, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, 'predict'):
                return self.adjusted_predict_call(self.base_element.predict, X, **kwargs)
            else:
                logger.error('BaseException. base Element should have function ' +
                               'predict.')
                raise BaseException('base Element should have function predict.')
        else:
            return X

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calls the predict function of the underlying base_element.

        Parameters:
            X:
                The array-like training and test data with shape=[N, D],
                where N is the number of samples and D is the number of features.

            **kwargs:
                Keyword arguments, passed to base_element.predict.

        Returns:
            Predictions values.

        """
        if self.batch_size == 0:
            return self.__predict(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict, X, **kwargs)

    def predict_proba(self, X, **kwargs):
        if self.batch_size == 0:
            return self.__predict_proba(X, **kwargs)
        else:
            return self.__batch_predict(self.__predict_proba, X, **kwargs)

    def __predict_proba(self, X: np.ndarray, **kwargs):
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
                # logger.error('BaseException. base Element should have "predict_proba" function.')
                # raise BaseException('base Element should have predict_proba function.')
                return None
        return X

    def __transform(self, X, y=None, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, 'transform'):
                return self.adjusted_delegate_call(self.base_element.transform, X, y, **kwargs)
            elif hasattr(self.base_element, 'predict'):
                return self.predict(X, **kwargs), y, kwargs
            else:
                logger.error('BaseException: transform-predict-mess')
                raise BaseException('transform-predict-mess')
        else:
            return X, y, kwargs

    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray, dict):
        """
        Calls transform on the base element.

        In case there is no transform method, calls predict.
        This is used if we are using an estimator as a preprocessing step.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the
                number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N], where N is
                the number of samples.

            **kwargs:
                Keyword arguments, passed to base_element.transform.

        Returns:
            (X, y) in transformed version and original kwargs.

        """
        if self.batch_size == 0:
            Xt, yt, kwargs = self.__transform(X, y, **kwargs)
        else:
            Xt, yt, kwargs = self.__batch_transform(X, y, **kwargs)
        if all(hasattr(data, "shape") for data in [X, Xt]) and all(len(data.shape) > 1 for data in [X, Xt]):
            self.reduce_dimension = (Xt.shape[1] < X.shape[1])
        return Xt, yt, kwargs

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray, dict):
        """
        Calls inverse_transform on the base element.

        When the dimension is preserved: transformers
        without inverse returns original input.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N
                is the number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N], where N is
                the number of samples.

            **kwargs:
                Keyword arguments, passed to base_element.transform.

        Raises:
            NotImplementedError:
                Thrown when there is a dimensional reduction but no inverse is defined.

        Returns:
            (X, y, kwargs) in back-transformed version.

        """
        if hasattr(self.base_element, 'inverse_transform'):
            # todo: check this
            X, y, kwargs = self.adjusted_delegate_call(self.base_element.inverse_transform, X, y, **kwargs)
        elif self.is_transformer and self.reduce_dimension:
            msg = "{} has no inverse_transform, but element reduce dimesions.".format(self.name)
            logger.error(msg)
            raise NotImplementedError(msg)
        return X, y, kwargs

    def __batch_transform(self, X, y=None, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            warning = "Cannot do batching on a single entity."
            logger.warning(warning)
            warnings.warn(warning)
            return self.__transform(X, y, **kwargs)

            # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        nr = PhotonDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PhotonDataHelper.chunker(nr, self.batch_size):
            batch_idx += 1

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PhotonDataHelper.split_data(X, y, kwargs, start, stop)

            actual_batch_size = PhotonDataHelper.find_n(X_batched)
            logger.debug(self.name + " is transforming batch " + str(batch_idx) + " with " + str(actual_batch_size)
                         + " items.")

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

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calls the score function on the base element.

        Parameters:
            X_test:
                Input test data to score on.

            y_test:
                Input true targets to score on.

        Returns:
            A goodness of fit measure or a likelihood of unseen data.

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

    @staticmethod
    def sanity_check_element_type_for_building_photon_pipes(pipe_element, type_of_self):
        if (not isinstance(pipe_element, PipelineElement) and not isinstance(pipe_element, PhotonNative)) or isinstance(pipe_element, Preprocessing):
            raise TypeError(str(type_of_self) + " only accepts PHOTON elements. Cannot add element of type " + str(type(pipe_element)))


class Branch(PipelineElement):
    """
     A substream of pipeline elements that is encapsulated, e.g. for parallelization.

     Example:
         ``` python
         from photonai.base import Branch
         from photonai.optimization import IntegerRange

         tree_qua_branch = Branch('tree_branch')
         tree_qua_branch += PipelineElement('QuantileTransformer', n_quantiles=100)
         tree_qua_branch += PipelineElement('DecisionTreeClassifier',
                                            {'min_samples_split': IntegerRange(2, 4)},
                                            criterion='gini')
         ```

     """
    def __init__(self, name: str, elements: List[PipelineElement] = None):
        """
        Initialize the object.

        Parameters:
            name:
                Name of the encapsulated item and/or
                summary of the encapsulated element`s functions.

            elements:
                List of PipelineElements added one after another to the Branch.

        """
        super().__init__(name, {}, test_disabled=False, disabled=False, base_element=True)

        # in case any of the children needs y or covariates we need to request them
        self.needs_y = True
        self.needs_covariates = True
        self.elements = []
        self.has_hyperparameters = True
        self.skip_caching = True
        self.identifier = "BRANCH:"

        # needed for caching on individual level
        self.fix_fold_id = False
        self.do_not_delete_cache_folder = False
        
        # add elements
        if elements:
            for element in elements:
                self.add(element)

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Calls the fit function on all underlying base elements.

        Parameters:
            X:
                The array-like input with shape=[N, D], where N is
                the number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to base_elements fit.

        Returns:
            Fitted self.

        """
        self.base_element = Branch.sanity_check_pipeline(self.base_element)
        return super().fit(X, y, **kwargs)

    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray, dict):
        """
        Calls the transform function on all underlying base elements.
        If _estimator_type is in ['classifier', 'regressor'], predict is called instead.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the
                number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to base_elements predict/transform.

        Returns:
            Transformed/Predicted data.

        """
        if self._estimator_type == 'classifier' or self._estimator_type == 'regressor':
            return super().predict(X), y, kwargs
        return super().transform(X, y, **kwargs)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calls the predict function on underlying base elements.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the
                number of samples and D is the number of features.

            **kwargs:
                Keyword arguments, passed to base_elements predict method.

        Returns:
            Prediction values.

        """
        return super().predict(X, **kwargs)

    def __iadd__(self, pipe_element: PipelineElement):
        """
        Add an element to the sub pipeline.

        Parameters:
            pipe_element:
                The PipelineElement to add, being either a transformer or an estimator.

        """
        super(Branch, self).__iadd__(pipe_element)
        self._prepare_pipeline()
        return self

    def add(self, pipe_element: PipelineElement):
        """
        Add an element to the sub pipeline.

        Parameters:
            pipe_element:
                The PipelineElement to add, being either a transformer or an estimator.

        """
        self.__iadd__(pipe_element)

    @staticmethod
    def prepare_photon_pipe(elements):
        pipeline_steps = list()
        for item in elements:
            pipeline_steps.append((item.name, item))
        return PhotonPipeline(pipeline_steps)

    @staticmethod
    def sanity_check_pipeline(pipe):
        if isinstance(pipe.elements[-1][1], CallbackElement):
            msg = "Last element of pipeline cannot be callback element, would be mistaken for estimator. Removing it."
            logger.warning(msg)
            warnings.warn(msg)
            del pipe.elements[-1]
        return pipe

    def _prepare_pipeline(self):
        """ Generates sklearn pipeline with all underlying elements """
        self._hyperparameters = {item.name: item.hyperparameters for item in self.elements
                                 if hasattr(item, 'hyperparameters')}

        if self.has_hyperparameters:
            self.generate_sklearn_hyperparameters()
        new_pipe = Branch.prepare_photon_pipe(self.elements)
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
        new_copy_of_me._random_state = self._random_state
        return new_copy_of_me

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that
        added can be optimized, not the container (self).
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
        Generates a dictionary according to the sklearn convention of
        element_name__parameter_name: parameter_value
        """
        self._hyperparameters = {}
        for element in self.elements:
            for attribute, value_list in element.hyperparameters.items():
                self._hyperparameters[self.name + '__' + attribute] = value_list

    def _check_hyper(self, BaseEstimator):
        pass

    @property
    def feature_importances_(self):
        if hasattr(self.elements[-1], 'feature_importances_'):
            return getattr(self.elements[-1], 'feature_importances_')


class Preprocessing(Branch):
    """
    Special kind of Branch.

    If a Preprocessing pipe is added to a PHOTONAI Hyperpipe,
    all transformers are applied to the data ONCE
    BEFORE cross validation starts in order to prepare the data.
    Every added element should be a transformer PipelineElement.

    Example:
        ``` python
        pre_proc = Preprocessing()
        pre_proc += PipelineElement('OneHotEncoder', sparse=False)
        my_pipe += pre_proc
        ```
        Some transformations should be performed bundled at the beginning.
        Here at the example of the OneHotEncoder. Due to the cross-validation split,
        some cateogries can no longer occur in any subsets.
        Therefore, a trained OneHotEncoding could fail on other subsets.
        By using the Preprocessing object, this effect can no longer appear.

    """
    def __init__(self):
        """Initialize the object."""
        super().__init__('Preprocessing')
        self.has_hyperparameters = False
        self.needs_y = True
        self.needs_covariates = True
        self._name = 'Preprocessing'
        self.is_transformer = True
        self.is_estimator = False

    def __iadd__(self, pipe_element: PipelineElement):
        """
        Add an element to the sub pipeline.

        Parameters:
            pipe_element:
                The transformer object to add.

        """
        if hasattr(pipe_element, "transform"):
            super(Preprocessing, self).__iadd__(pipe_element)
            if len(pipe_element.hyperparameters) > 0:
                raise ValueError("A preprocessing transformer must not have any hyperparameter "
                                 "because it is not part of the optimization and cross validation procedure")

        else:
            raise ValueError("Pipeline Element must have transform function")
        return self

    def predict(self, data, **kwargs):
        warnings.warn("There is no predict function of the preprocessing pipe, it is a transformer only.")
        pass

    @property
    def _estimator_type(self):
        return


class Stack(PipelineElement):
    """
    Creates a vertical stacking/parallelization of pipeline items.

    The object acts as a single PipelineElement and encapsulates
    several vertically stacked other PipelineElements, each
    child receiving the same input data. The data is iteratively
    distributed to all children, the results are collected
    and horizontally concatenated.

    Example:
        ``` python
        tree = PipelineElement('DecisionTreeClassifier')
        svc = PipelineElement('LinearSVC')

        my_pipe += Stack('final_stack', [tree, svc], use_probabilities=True)
        ```

    """
    def __init__(self, name: str, elements: List[PipelineElement] = None, use_probabilities: bool = False):
        """
        Creates a new Stack element.
        Collects all possible hyperparameter combinations of the children.

        Parameters:
            name:
                Give the pipeline element a name.

            elements:
                List of pipeline elements that should run in parallel.

            use_probabilities:
                For a stack that includes estimators you can choose whether
                predict or predict_proba is called for all estimators.
                In case only some implement predict_proba, predict
                is called for the remaining estimators.

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
        self.identifier = "STACK:"
        self.use_probabilities = use_probabilities

    def __iadd__(self, item: PipelineElement):
        """
        Add a new element to the stack.
        Generate sklearn hyperparameter names in order
        to set the item's hyperparameters in the optimization process.

        Parameters:
            item:
                The Element that should be stacked and will run in a
                vertical parallelization in the original pipe.

        """
        self.check_if_needs_y(item)
        super(Stack, self).__iadd__(item)

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

    def add(self, item: PipelineElement):
        """
        Add a new element to the stack.
        Generate sklearn hyperparameter names in order
        to set the item's hyperparameters in the optimization process.

        Parameters:
            item:
                The Element that should be stacked and will run in a
                vertical parallelization in the original pipe.

        """
        self.__iadd__(item)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added
        can be optimized, not the container (self).
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
        """Find the particular child and distribute the params to it"""
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

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Calls fit iteratively on every child.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the
                number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to base_elements fit.

        Returns:
            Fitted self.

        """
        for element in self.elements:
            # Todo: parallellize fitting
            element.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calls the predict function on underlying base elements.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the
                number of samples and D is the number of features.

            **kwargs:
                Keyword arguments, passed to base_elements predict.

        Returns:
            Prediction values.

        """
        if not self.use_probabilities:
            return self._predict(X, **kwargs)
        else:
            return self.predict_proba(X, **kwargs)

    def _predict(self, X: np.ndarray, **kwargs):
        """Iteratively calls predict on every child."""
        # Todo: strategy for concatenating data from different pipes
        # todo: parallelize prediction
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict(X, **kwargs)
            predicted_data = PhotonDataHelper.stack_data_horizontally(predicted_data, element_transform)
        return predicted_data

    def predict_proba(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Predict probabilities for every pipe element and stack them together.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the number
                of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, not used yet.

        Returns:
            Probability values.

        """
        predicted_data = np.array([])
        for element in self.elements:
            element_transform = element.predict_proba(X)
            if element_transform is None:
                element_transform = element.predict(X)
            predicted_data = PhotonDataHelper.stack_data_horizontally(predicted_data, element_transform)
        return predicted_data

    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray, dict):
        """
        Calls transform on every child.

        If the encapsulated child is a hyperpipe, also calls predict on the last element in the pipeline.

        Parameters:
            X:
                The array-liketraining with shape=[N, D] and test data,
                where N is the number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to base_elements transform.

        Returns:
            Prediction values.

        """
        transformed_data = np.array([])
        for element in self.elements:
            # if it is a hyperpipe with a final estimator, we want to use predict:
            element_transform, _, _ = element.transform(X, y, **kwargs)
            transformed_data = PhotonDataHelper.stack_data_horizontally(transformed_data, element_transform)

        return transformed_data, y, kwargs

    def copy_me(self):
        ps = Stack(self.name)
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps.base_element = self.base_element
        ps._random_state = self._random_state
        return ps

    def inverse_transform(self, X, y=None, **kwargs):
        raise NotImplementedError("Inverse Transform is not yet implemented for a Stacking Element in PHOTON")

    @property
    def _estimator_type(self):
        return None

    def _check_hyper(self,BaseEstimator):
        pass

    @property
    def feature_importances_(self):
        return


class Switch(PipelineElement):
    """
    This class encapsulates several PipelineElements that
    belong at the same step of the pipeline, competing for
    being the best choice.

    If for example you want to find out if Preprocessing A
    or Preprocessing B is better at this position in the pipe.
    Or you want to test if a random forest outperforms the good old SVM.

    ATTENTION: This class is a construct that may be convenient
    but is not suitable for any complex optimizations.
    Currently optimization works for grid_search, random search and
    smac and the specializes switch optimizer.

    Example:
        ``` python
        from photonai.base import PipelineElement, Switch
        from photonai.optimization import IntegerRange
        # Estimator Switch
        svm = PipelineElement('SVC',
                              hyperparameters={'kernel': ['rbf', 'linear']})

        tree = PipelineElement('DecisionTreeClassifier',
                               hyperparameters={'min_samples_split': IntegerRange(2, 5),
                                                'min_samples_leaf': IntegerRange(1, 5),
                                                'criterion': ['gini', 'entropy']})

        my_pipe += Switch('EstimatorSwitch', [svm, tree])
        ```

    """

    def __init__(self, name: str, elements: List[PipelineElement] = None, estimator_name: str = ''):
        """
        Creates a new Switch object and generated the hyperparameter combination grid.

        Parameters:
            name:
                How the element is called in the pipeline.

            elements:
                The competing pipeline elements.

            estimator_name:
                -

        """
        self._name = name
        self.initial_name = self._name
        self.sklearn_name = self.name + "__current_element"
        self._hyperparameters = {}
        self._current_element = (1, 1)
        self.pipeline_element_configurations = []
        self.base_element = None
        self.disabled = False
        self.test_disabled = False
        self.batch_size = 0
        self.estimator_name = estimator_name

        self.needs_y = True
        self.needs_covariates = True
        # we assume we test models against each other, but only guessing
        self.is_estimator = True
        self.is_transformer = True
        self.identifier = "SWITCH:"
        self._random_state = False

        self.elements_dict = {}

        if elements:
            self.elements = elements
            self.generate_private_config_grid()
            for p_element in elements:
                self.elements_dict[p_element.name] = p_element
        else:
            self.elements = []

    def __iadd__(self, pipeline_element: PipelineElement):
        """
        Add a new estimator or transformer object to the switch container.
        All items change their positions during testing.

        Parameters:
            pipeline_element:
                Item that should be tested against other
                competing elements at that position in the pipeline.

        """
        super(Switch, self).__iadd__(pipeline_element)
        self.elements_dict[pipeline_element.name] = pipeline_element
        self.generate_private_config_grid()
        return self

    def add(self, pipeline_element: PipelineElement):
        """
        Add a new estimator or transformer object to the switch container.
        All items change their positions during testing.

        Parameters:
            pipeline_element:
                Item that should be tested against other
                competing elements at that position in the pipeline.

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

            if hasattr(pipe_element, 'generate_config_grid'):
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
        The optimization process sees the amount of possible
        combinations and chooses one of them. Then this class activates
        the belonging element and prepared the element with the
        particular chosen configuration.

        """
        config_nr = None
        config = None
        self.estimator_name = ''
        # copy dict for adaptations
        params = dict(kwargs)

        # in case we are operating with grid search
        if self.sklearn_name in params:
            config_nr = params[self.sklearn_name]
        elif 'current_element' in params:
            config_nr = params['current_element']

        if "estimator_name" in kwargs:
            self.estimator_name = params["estimator_name"]
            del params["estimator_name"]
            self.base_element = self.elements_dict[self.estimator_name]

        if params is not None:
            config = params

        # todo: raise Warning that Switch could not identify which estimator to set when estimator
        #  has no params to optimize

        # in case we are operating with grid search or any derivates
        if config_nr is not None:

            if not isinstance(config_nr, (tuple, list)):
                logger.error('ValueError: current_element must be of type Tuple')
                raise ValueError('current_element must be of type Tuple')

            # grid search hack
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]
        # if we don't use the specialized switch optimizer
        # we need to identify the element to activate by checking for which element the optimizer gave params
        elif not self.estimator_name:
            # ugly hack because subscription is somehow not possible, we use the for loop but break
            for kwargs_key, kwargs_value in params.items():
                first_element_name = kwargs_key.split("__")[0]
                self.base_element = self.elements_dict[first_element_name]
                break

        # so now the element to be activated is found and taken care of,
        # let's move on to give the base element the config to set
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
        ps._random_state = self._random_state
        for element in self.elements:
            new_element = element.copy_me()
            ps += new_element
        ps._current_element = self._current_element
        return ps

    def prettify_config_output(self, config_name, config_value, return_dict=False) -> str:
        """
        Makes the sklearn configuration dictionary human readable.

        Returns:
            Configuration as prettified string or configuration as
            dict with prettified keys.

        """
        if isinstance(config_value, tuple):
            output = self.pipeline_element_configurations[config_value[0]][config_value[1]]
            if not output:
                if return_dict:
                    return {self.elements[config_value[0]].name: None}
                else:
                    return self.elements[config_value[0]].name
            else:
                if return_dict:
                    return output
                return str(output)
        else:
            return super(Switch, self).prettify_config_output(config_name, config_value)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Union[np.ndarray, None]:
        """
        Predict probabilities. Base element needs predict_proba()
        function, otherwise return None.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N is the number
                of samples and D is the number of features.

            **kwargs:
                Keyword arguments, not in use yet.

        Returns:
            Probabilities.

        """
        if not self.disabled:
            if hasattr(self.base_element.base_element, 'predict_proba'):
                return self.base_element.predict_proba(X)
            else:
                return None
        return X

    def _check_hyper(self,BaseEstimator):
        pass

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Calls inverse_transform on the base element.

        For a dimension preserving transformer without inverse,
        the value is returned untreated.

        Parameters:
            X:
                The array-like data with shape=[N, D], where N
                is the number of samples and D is the number of features.

            y:
                The truth array-like values with shape=[N], where N is
                the number of samples.

            **kwargs:
                Keyword arguments, passed to base_element.transform.

        Returns:
            (X, y, kwargs) in back-transformed version if possible.

            """
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
            return

    @property
    def feature_importances_(self):
        if hasattr(self.base_element, 'feature_importances_'):
            return getattr(self.base_element, 'feature_importances_')


class DataFilter(BaseEstimator, PhotonNative):
    """
    Helper Class to split the data e.g. for stacking.
    """
    def __init__(self, indices):
        self.name = 'DataFilter'
        self.hyperparameters = {}
        self.indices = indices
        self.needs_covariates = False
        self.needs_y = False
        self.disabled = False

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
        return


class CallbackElement(PhotonNative):

    def __init__(self, name, delegate_function, method_to_monitor='transform'):

        self.needs_covariates = True
        self.needs_y = True
        self.name = name
        # todo: check if delegate function accepts X, y, kwargs
        self.delegate_function = delegate_function
        self.method_to_monitor = method_to_monitor
        self.hyperparameters = {}
        self.is_transformer = True
        self.is_estimator = False
        self.disabled = False

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

    @property
    def feature_importances_(self):
        return


class ParallelBranch(Branch):
    """
    A substream of elements, that do not need fit() but can instantly use transform on a single subject level,
    and therefore can be computed in parallel.

    """

    def __init__(self, name: str, nr_of_processes: int = 1):
        """
        Initialize the object.

        Parameters:
            name:
                Name of the parallelizable pipeline branch.

            nr_of_processes:
                Number of process run in parallel.

        """
        Branch.__init__(self, name)

        self._nr_of_processes = 1
        self.local_cluster = None
        self.client = None
        self.nr_of_processes = nr_of_processes

        self.has_hyperparameters = True
        self.needs_y = False
        self.needs_covariates = True
        self.current_config = None

    def __del__(self):
        if self.local_cluster is not None:
            self.local_cluster.close()

    def fit(self, X, y=None, **kwargs):
        # do nothing here!!
        return self

    @property
    def nr_of_processes(self):
        return self._nr_of_processes
    # Todo : !
    # @classmethod
    # def set_local_cluster(cls, nr_of_processes):
    #     cls.local_cluster =

    @nr_of_processes.setter
    def nr_of_processes(self, value):
        self._nr_of_processes = value
        if self._nr_of_processes > 1:
            if self.local_cluster is None:
                self.local_cluster = Client(threads_per_worker=1,
                                            n_workers=self.nr_of_processes,
                                            processes=False)
            else:
                self.local_cluster.n_workers = self.nr_of_processes
        else:
            self.local_cluster = None

    def __iadd__(self, pipe_element: PipelineElement):
        """
        Add an element to the ParallelBranch.

        Parameters:
            pipe_element:
                The transformer object to add.

        """
        self.elements.append(pipe_element)
        self._prepare_pipeline()

        return self

    def transform(self, X, y=None, **kwargs):

        if self.base_element.cache_folder is not None:
            # make sure we cache individually
            self.base_element.single_subject_caching = True
            self.base_element.caching = True
        if self.nr_of_processes > 1:

            if self.base_element.cache_folder is not None:
                # at first apply the transformation on several cores, everything gets written to the cache,
                # so the next step only has to reload the data ...
                self.apply_transform_parallelized(X)
            else:
                logger.error("Cannot use parallelization without a cache folder specified in the hyperpipe."
                               "Using single core instead")

            logger.debug('ParallelBranch ' + self.name + ' is collecting data from the different cores...')
        X_new, _, _ = self.base_element.transform(X)

        return X_new, y, kwargs

    def set_params(self, **kwargs):
        self.current_config = kwargs
        super(ParallelBranch, self).set_params(**kwargs)

    def copy_me(self):
        new_copy = super().copy_me()
        new_copy.base_element.current_config = self.base_element.current_config
        new_copy.base_element.single_subject_caching = True
        new_copy.base_element.cache_folder = self.base_element.cache_folder
        new_copy.local_cluster = self.local_cluster
        new_copy.nr_of_processes = self.nr_of_processes

        # todo: clarify this with Ramona
        new_copy.do_not_delete_cache_folder = True

        return new_copy

    def inverse_transform(self, X, y=None, **kwargs):
        for transform in self.elements[::-1]:
            if hasattr(transform, 'inverse_transform'):
                X, y, kwargs = transform.inverse_transform(X, y, **kwargs)
            else:
                return X, y, kwargs
        return X, y, kwargs

    @staticmethod
    def parallel_application(pipe_copy, data):
        pipe_copy.transform(data)
        return

    def apply_transform_parallelized(self, X: np.ndarray):
        """
        Apply transformation in parallel.

        Parameters:
            X:
                The data to which the delegate should be applied in parallel.
        """

        if self.nr_of_processes > 1:

            jobs_to_do = list()

            # distribute the data equally to all available cores
            number_of_items_to_process = PhotonDataHelper.find_n(X)
            number_of_items_for_each_core = int(np.ceil(number_of_items_to_process / self.nr_of_processes))
            logger.info('ParallelBranch ' + self.name +
                         ': Using ' + str(self.nr_of_processes) + ' cores calculating ' + str(number_of_items_for_each_core)
                         + ' items each')
            for start, stop in PhotonDataHelper.chunker(number_of_items_to_process, number_of_items_for_each_core):
                X_batched, _, _ = PhotonDataHelper.split_data(X, None, {}, start, stop)

                # copy my pipeline
                new_pipe_mr = self.copy_me()
                new_pipe_copy = new_pipe_mr.base_element
                new_pipe_copy.cache_folder = self.base_element.cache_folder
                new_pipe_copy.skip_loading = True
                new_pipe_copy._parallel_use = True

                del_job = dask.delayed(ParallelBranch.parallel_application)(new_pipe_copy, X_batched)
                jobs_to_do.append(del_job)

            dask.compute(*jobs_to_do)
