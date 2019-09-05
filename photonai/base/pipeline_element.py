import importlib
import inspect

from sklearn.base import BaseEstimator
from sklearn.model_selection._search import ParameterGrid

from photonai.base import PhotonRegister, Switch, Branch, Preprocessing
from photonai.base.helper import PhotonDataHelper
from photonai.photonlogger import Logger
from photonai.optimization.config_grid import create_global_config_dict


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
    ELEMENT_DICTIONARY = PhotonRegister().get_package_info()

    def __init__(self, name, hyperparameters: dict=None, test_disabled: bool=False,
                 disabled: bool =False, base_element=None, batch_size=0, **kwargs):
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

        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized --> sklearn: get_params.keys
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.test_disabled = test_disabled
        self._sklearn_disabled = self.name + '__disabled'
        self._hyperparameters = hyperparameters

        # check if hyperparameters are members of the class
        self._check_hyper(BaseEstimator)

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

    def _check_hyper(self,BaseEstimator):
        # check if hyperparameters are members of the class
        not_supp_hyper = list(
            set([key.split("__")[-1] for key in self._hyperparameters.keys() if key.split("__")[-1]!="disabled"]) - set(BaseEstimator.get_params(self.base_element).keys()))
        if not_supp_hyper:
            Logger().error(
                'ValueError: Set of hyperparameters are not valid, check hyperparameters:' + str(not_supp_hyper))
            raise ValueError(
                'ValueError: Set of hyperparameters are not valid, check hyperparameters:' + str(not_supp_hyper))

    def copy_me(self):
        # TODO !!!!!!!
        if self.name in self.ELEMENT_DICTIONARY:
            copy = PipelineElement(self.name, self.hyperparameters, **self.kwargs)
        else:
            # handle custom elements
            copy = PipelineElement.create(self.name, self.base_element, hyperparameters=self.hyperparameters, **self.kwargs)
        if self.current_config is not None:
            copy.set_params(**self.current_config)
        return copy
        # if hasattr(self.base_element, 'copy_me'):
        #     # new_base_element = self.base_element.copy_me()

        #     return PipelineElement(self.name, self.hyperparameters, **self.kwargs)
        # else:
        #     return deepcopy(self)

    @classmethod
    def create(cls, name, base_element, hyperparameters: dict, test_disabled=False, disabled=False, **kwargs):
        """
        Takes an instantiated object and encapsulates it into the PHOTON structure,
        add the disabled function and attaches information about the hyperparameters that should be tested
        """
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

    # def fit_predict(self, data, targets):
    #     if not self.disabled:
    #         return self.base_element.fit_predict(data, targets)
    #     else:
    #         return data

    def __transform(self, X, y=None, **kwargs):
        if not self.disabled:
            if hasattr(self.base_element, 'transform'):
                return self.adjusted_delegate_call(self.base_element.transform, X, y, **kwargs)
            elif hasattr(self.base_element, 'predict'):
                # Logger().warn("used prediction instead of transform " + self.name)
                # raise Warning()
                # todo: here, I used delegate call instead to differentiate between estimator that need kwargs and those which don't
                return self.predict(X, **kwargs), y, kwargs
                #return self.base_element.predict(X), y, kwargs

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
                    # i.e. if we change the number of samples, we also need to apply that change to all kwargs
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
