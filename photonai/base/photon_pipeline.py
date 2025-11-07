import datetime
import os

import numpy as np
import warnings
from sklearn.utils.metaestimators import _BaseComposition

from photonai.helper.helper import PhotonDataHelper
from photonai.photonlogger.logger import logger


class PhotonPipeline(_BaseComposition):

    def __init__(self, elements, random_state=False):
        self.elements = elements
        self.random_state = random_state
        self.current_config = None
        # caching stuff
        self.caching = False
        self._fold_id = None
        self._cache_folder = None
        self.time_monitor = {'fit': [], 'transform_computed': [], 'transform_cached': [], 'predict': []}
        self.cache_man = None

        # helper for single subject caching
        self._single_subject_caching = False
        self._fix_fold_id = False
        self._do_not_delete_cache_folder = False
        self._parallel_use = False

        # helper for optimum pipe
        self._meta_information = None

        # used in parallelization
        self.skip_loading = False


    def set_lock(self, lock):
        self.cache_man.lock = lock

    @property
    def single_subject_caching(self):
        return self._single_subject_caching

    @single_subject_caching.setter
    def single_subject_caching(self, value: bool):
        if value:
            self._fix_fold_id = True
            self._do_not_delete_cache_folder = True
        else:
            self._fix_fold_id = False
            self._do_not_delete_cache_folder = False
        self._single_subject_caching = value

    @property
    def fold_id(self):
        return self._fold_id

    @fold_id.setter
    def fold_id(self, value):
        if value is None:
            self._fold_id = ''
            # we dont need group-wise caching if we have no inner fold id
            self.caching = False
            self.cache_man = None
        else:
            if self._fix_fold_id:
                self._fold_id = "fixed_fold_id"
            else:
                self._fold_id = str(value)
            self.caching = True
            self.cache_man = CacheManager(self._fold_id, self.cache_folder, self._parallel_use,
                                          self._single_subject_caching)

    @property
    def cache_folder(self):
        return self._cache_folder

    @cache_folder.setter
    def cache_folder(self, value):

        if not self._do_not_delete_cache_folder:
            self._cache_folder = value
        else:
            if isinstance(value, str) and not value.endswith("DND"):
                self._cache_folder = value + "DND"
            else:
                self._cache_folder = value

        if isinstance(self._cache_folder, str):
            self.caching = True
            if not os.path.isdir(self._cache_folder):
                os.makedirs(self._cache_folder)
            self.cache_man = CacheManager(self._fold_id, self.cache_folder, self._parallel_use, self._single_subject_caching)
        else:
            self.caching = False

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('elements', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        if self.current_config is not None and len(self.current_config) > 0:
            if kwargs is not None and len(kwargs) == 0:
                raise ValueError("Pipeline cannot set parameters to elements with an emtpy dictionary. Old values persist")
        self.current_config = kwargs
        self._set_params('elements', **kwargs)

        return self

    def _validate_elements(self):
        names, estimators = zip(*self.elements)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None:
                continue
            if not (hasattr(t, "fit") or not hasattr(t, "transform")):
                raise TypeError("All intermediate elements should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if estimator is not None and not hasattr(estimator, "fit"):
            raise TypeError("Last step of Pipeline should implement fit. "
                            "'%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    def fit(self, X, y=None, **kwargs):

        self._validate_elements()
        X, y, kwargs = self._caching_fit_transform(X, y, kwargs, fit=True)

        if self._final_estimator is not None:
            logger.debug('PhotonPipeline: Fitting ' + self._final_estimator.name)
            fit_start_time = datetime.datetime.now()
            self._final_estimator.fit(X, y, **kwargs)
            n = PhotonDataHelper.find_n(X)
            fit_duration = (datetime.datetime.now() - fit_start_time).total_seconds()
            self.time_monitor['fit'].append((self.elements[-1][0], fit_duration, n))
        return self

    def check_for_numpy_array(self, list_object):
        # be compatible to list of (image-) files
        if isinstance(list_object, list):
            return np.asarray(list_object)
        else:
            return list_object

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on every step that offers a transform function
        including the last step if it has the transformer flag,
        and excluding the last step if it has the estimator flag but no transformer flag.

        Returns transformed X, y and kwargs
        """
        X, y, kwargs = self._caching_fit_transform(X, y, kwargs)

        if self._final_estimator is not None:
            if self._estimator_type is None:
                logger.debug('PhotonPipeline: Transforming data with ' + self._final_estimator.name)
                X, y, kwargs = self._final_estimator.transform(X, y, **kwargs)

        return X, y, kwargs

    def _do_timed_fit_transform(self, name, transformer, fit, X, y, **kwargs):

        n = PhotonDataHelper.find_n(X)
        if fit:
            logger.debug('PhotonPipeline: Fitting ' + transformer.name)
            fit_start_time = datetime.datetime.now()
            transformer.fit(X, y, **kwargs)
            fit_duration = (datetime.datetime.now() - fit_start_time).total_seconds()
            self.time_monitor['fit'].append((name, fit_duration, n))

        logger.debug('PhotonPipeline: Transforming data with ' + transformer.name)
        transform_start_time = datetime.datetime.now()
        X, y, kwargs = transformer.transform(X, y, **kwargs)
        transform_duration = (datetime.datetime.now() - transform_start_time).total_seconds()
        self.time_monitor['transform_computed'].append((name, transform_duration, n))
        return X, y, kwargs

    def _caching_fit_transform(self, X, y, kwargs, fit=False):
        for num, (name, transformer) in enumerate(self.elements[:-1]):
            X, y, kwargs = self._do_timed_fit_transform(name, transformer, fit, X, y, **kwargs)

            # always work with numpy arrays to avoid checking for shape attribute
            X = self.check_for_numpy_array(X)
            y = self.check_for_numpy_array(y)

        return X, y, kwargs

    def predict(self, X, training=False, **kwargs):
        """
        Transforms the data for every step that offers a transform function
        and then calls the estimator with predict on transformed data.
        It returns the predictions made.

        In case the last step is no estimator, it returns the transformed data.
        """

        # first transform
        if not training:
            X, _, kwargs = self.transform(X, y=None, **kwargs)

        # then call predict on final estimator
        if self._final_estimator is not None:
            if self._final_estimator.is_estimator:
                logger.debug('PhotonPipeline: Predicting with ' + self._final_estimator.name + ' ...')
                predict_start_time = datetime.datetime.now()
                y_pred = self._final_estimator.predict(X, **kwargs)
                predict_duration = (datetime.datetime.now() - predict_start_time).total_seconds()
                n = PhotonDataHelper.find_n(X)
                self.time_monitor['predict'].append((self.elements[-1][0], predict_duration, n))
                return y_pred
            else:
                return X
        else:
            return None

    def predict_proba(self, X, training: bool=False, **kwargs):
        if not training:
            X, _, kwargs = self.transform(X, y=None, **kwargs)

        if self._final_estimator is not None:
            if self._final_estimator.is_estimator:
                if hasattr(self._final_estimator, "predict_proba"):
                    if hasattr(self._final_estimator, 'needs_covariates'):
                        if self._final_estimator.needs_covariates:
                            return self._final_estimator.predict_proba(X, **kwargs)
                        else:
                            return self._final_estimator.predict_proba(X)
                    else:
                        return self._final_estimator.predict_proba(X)

        raise NotImplementedError("The final estimator does not have a predict_proba method")

    def fit_transform(self, X, y=None, **kwargs):
        # return self.fit(X, y, **kwargs).transform(X, y, **kwargs)
        raise NotImplementedError('fit_transform not yet implemented in PHOTON Pipeline')

    def fit_predict(self, X, y=None, **kwargs):
        raise NotImplementedError('fit_predict not yet implemented in PHOTON Pipeline')

    def copy_me(self):
        pipeline_steps = []
        for item_name, item in self.elements:
            cpy = item.copy_me()
            if isinstance(cpy, list):
                for new_step in cpy:
                    pipeline_steps.append((new_step.name, new_step))
            else:
                pipeline_steps.append((cpy.name, cpy))
        new_pipe = PhotonPipeline(pipeline_steps)
        new_pipe.random_state = self.random_state
        return new_pipe

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Transforms the data for every step that offers a transform function
        and then calls the estimator with predict on transformed data.
        It returns the predictions made.

        In case the last step is no estimator, it returns the transformed data.

        Parameters:
            X:
                Test samples.

            y:
                True values for `X`.

            kwargs:
                Passed to final_estimator.score(), e.g. sample_weight possible.

        Returns:
            Score value.
        """

        X, y, kwargs = self.transform(X, y=y, **kwargs)

        # call score on final estimator
        if self._final_estimator is not None:
            if self._final_estimator.is_estimator:
                return self._final_estimator.score(X, y, **kwargs)

        msg = "It is not possible to run the score method without matching final_estimator. " \
              "Make sure that the last element is an estimator with integrated score function."
        logger.error(msg)
        raise ValueError(msg)

    @property
    def named_steps(self):
        return dict(self.elements)

    @property
    def _final_estimator(self):
        return self.elements[-1][1]

    @property
    def _estimator_type(self):
        return getattr(self._final_estimator, '_estimator_type')

    def clear_cache(self):
        if self.cache_man is not None:
            self.cache_man.clear_cache()

    def add_preprocessing(self, preprocessing):
        if preprocessing:
            self.elements.insert(0, (preprocessing.name, preprocessing))

    @property
    def feature_importances_(self):
        return self.elements[-1][1].feature_importances_
