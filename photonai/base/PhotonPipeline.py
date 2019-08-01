from sklearn.utils.metaestimators import _BaseComposition
import numpy as np
import os
import pickle
import uuid
import shutil
from ..photonlogger.Logger import Logger


class PhotonPipeline(_BaseComposition):

    def __init__(self, steps):
        self.steps = steps

        self.current_config = None
        self._fold_id = None
        self.fix_fold_id = False
        self._cache_folder = None
        self.do_not_delete_cache_folder = False
        self.caching = False

        self.cache_man = CacheManager(self._fold_id, self.cache_folder)

    @property
    def fold_id(self):
        return self._fold_id

    @fold_id.setter
    def fold_id(self, value: uuid.UUID):
        if self.fix_fold_id:
            self._fold_id = "fixed_fold_id"
        else:
            self._fold_id = str(value)

    @property
    def cache_folder(self):
        return self._cache_folder

    @cache_folder.setter
    def cache_folder(self, value):
        if not self.do_not_delete_cache_folder:
            self._cache_folder = value
        else:
            if isinstance(value, str):
                self._cache_folder = value + "DND"
            else:
                self._cache_folder = value

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
        return self._get_params('steps', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self.current_config = kwargs
        self._set_params('steps', **kwargs)

        return self

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None:
                continue
            if not (hasattr(t, "fit") or not hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if estimator is not None and not hasattr(estimator, "fit"):
            raise TypeError("Last step of Pipeline should implement fit. "
                            "'%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    def fit(self, X, y=None, **kwargs):

        self._validate_steps()
        X, y, kwargs = self._caching_fit_transform(X, y, kwargs, fit=True)

        if self._final_estimator is not None:
            self._final_estimator.fit(X, y, **kwargs)

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
            if self._final_estimator.is_transformer and not self._final_estimator.is_estimator:
                if self.caching and len(self.current_config) > 0:
                    X, y, kwargs = self.load_or_save_cached_data(self._final_estimator.name, X, y, kwargs, self._final_estimator)
                else:
                    X, y, kwargs = self._final_estimator.transform(X, y, **kwargs)

        if self.caching:
            self.cache_man.save_cache_index()

        return X, y, kwargs

    def load_or_save_cached_data(self, name, X, y, kwargs, transformer, fit=False):
        cached_result = self.cache_man.load_cached_data(name)
        if cached_result is None:
            if fit:
                transformer.fit(X, y, **kwargs)
            X, y, kwargs = transformer.transform(X, y, **kwargs)
            self.cache_man.save_data_to_cache(name, (X, y, kwargs))
        else:
            X, y, kwargs = cached_result[0], cached_result[1], cached_result[2]
        return X, y, kwargs

    def _caching_fit_transform(self, X, y, kwargs, fit=False):

        if self.caching:
            # update infos, just in case
            self.cache_man.hash = self._fold_id
            self.cache_man.cache_folder = self.cache_folder
            self.cache_man.prepare([name for name, e in self.steps], X, self.current_config)
            last_cached_item = None

        # all steps except the last one
        num_steps = len(self.steps) - 1

        for num, (name, transformer) in enumerate(self.steps[:-1]):
            if not self.caching or self.current_config is None or \
                    (hasattr(transformer, 'skip_caching') and transformer.skip_caching):
                if fit:
                    transformer.fit(X, y, **kwargs)
                X, y, kwargs = transformer.transform(X, y, **kwargs)
            else:
                # load data when the first item occurs that needs new calculation
                if self.cache_man.check_cache(name):
                    # as long as we find something cached, we remember what it was
                    last_cached_item = name
                    # if it is the last step, we need to load the data now
                    if num + 1 == num_steps:
                        X, y, kwargs = self.load_or_save_cached_data(last_cached_item, X, y, kwargs, transformer, fit)
                else:
                    if last_cached_item is not None:
                        # we load the cached data when the first transformation on this data is upcoming
                        X, y, kwargs = self.load_or_save_cached_data(last_cached_item, X, y, kwargs, transformer, fit)
                    X, y, kwargs = self.load_or_save_cached_data(name, X, y, kwargs, transformer, fit)

            # always work with numpy arrays to avoid checking for shape attribute
            X = self.check_for_numpy_array(X)
            y = self.check_for_numpy_array(y)

        if self.caching:
            self.cache_man.save_cache_index()

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
                y_pred = self._final_estimator.predict(X, **kwargs)
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

    def inverse_transform(self, X, y=None, **kwargs):
        # simply use X to apply inverse_transform
        # does not work on any transformers changing y or kwargs!
        for name, transform in self.steps[::-1]:
            if hasattr(transform, 'inverse_transform'):
                X, y, kwargs = transform.inverse_transform(X, y, **kwargs)
        return X, y, kwargs

    def fit_transform(self, X, y=None, **kwargs):
        # return self.fit(X, y, **kwargs).transform(X, y, **kwargs)
        raise NotImplementedError('fit_transform not yet implemented in PHOTON Pipeline')

    def fit_predict(self, X, y=None, **kwargs):
        raise NotImplementedError('fit_predict not yet implemented in PHOTON Pipeline')


    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]


class CacheManager:

    def __init__(self, _hash=None, cache_folder=None):
        self._hash = _hash
        self.cache_folder = cache_folder

        self.pipe_order = None
        self.cache_index = None
        self.state = None
        self.cache_file_name = None

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        if not isinstance(value, str):
            self._hash = str(value)
        else:
            self._hash = value

    class State:
        def __init__(self, config=None, nr_items=None,
                     first_data_hash=None, first_data_str: str = None):
            self.config = config
            self.nr_items = nr_items
            self.first_data_hash = first_data_hash
            self.first_data_str = first_data_str

    def prepare(self, pipe_elements, X, config):

        cache_name = 'photon_cache_index.p'
        self.cache_file_name = os.path.join(self.cache_folder, cache_name)

        if os.path.isfile(self.cache_file_name):
            with open(self.cache_file_name, 'rb') as f:
                self.cache_index = pickle.load(f)
        else:
            self.cache_index = {}
        self.pipe_order = pipe_elements

        first_item = X[0]
        first_item_hash = hash(str(first_item))
        self.state = CacheManager.State(config=config, first_data_hash=first_item_hash)

        if isinstance(X, np.ndarray):
            self.state.nr_items = X.shape[0]
            self.state.first_data_str = str(self.state.first_data_hash)
        else:
            self.state.nr_items = len(X)
            if isinstance(first_item, str):
                self.state.first_data_str = first_item
            else:
                self.state.first_data_str = str(self.state.first_data_hash)

    def _find_config_for_element(self, pipe_element_name):
        relevant_keys = list()
        for item in self.pipe_order:
            if item != pipe_element_name:
                relevant_keys.append(item)
            elif item == pipe_element_name:
                relevant_keys.append(item)
                break

        relevant_dict = dict()
        if self.state.config is not None and len(self.state.config) > 0:
            for key_name, key_value in self.state.config.items():
                key_name_list = key_name.split("__")
                if len(key_name_list) > 0:
                    item_name = key_name_list[0]
                else:
                    item_name = key_name

                if item_name in relevant_keys:
                    relevant_dict[key_name] = key_value

        return hash(frozenset(relevant_dict.items()))

    def load_cached_data(self, pipe_element_name):

        if pipe_element_name in ["SmoothImages", "ResampleImages", "BrainAtlas"]:
            debug = True

        config_hash = self._find_config_for_element(pipe_element_name)
        cache_query = (pipe_element_name, self.hash, config_hash, self.state.nr_items, self.state.first_data_hash)
        if cache_query in self.cache_index:
            Logger().debug("Loading data from cache for " + pipe_element_name + ": "
                           + str(self.state.nr_items) + " items " + self.state.first_data_str
                           + " - " + str(self.state.config))
            with open(self.cache_index[cache_query], 'rb') as f:
                (X, y, kwargs) = pickle.load(f)
            return X, y, kwargs
        return None

    def check_cache(self, pipe_element_name):
        config_hash = self._find_config_for_element(pipe_element_name)
        cache_query = (pipe_element_name, self.hash, config_hash, self.state.nr_items, self.state.first_data_hash)
        if cache_query in self.cache_index:
            return True
        else:
            return False

    def save_data_to_cache(self, pipe_element_name, data):
        config_hash = self._find_config_for_element(pipe_element_name)
        filename = os.path.join(self.cache_folder, str(uuid.uuid4()) + ".p")
        self.cache_index[(pipe_element_name, self.hash, config_hash, self.state.nr_items,
                          self.state.first_data_hash)] = filename
        Logger().debug("Saving data to cache for " + pipe_element_name + ": " + str(self.state.nr_items) + " items "
                       + self.state.first_data_str + " - " + str(self.state.config))
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def save_cache_index(self):
        with open(self.cache_file_name, 'wb') as f:
            pickle.dump(self.cache_index, f)

    def clear_cache(self):
        CacheManager.clear_cache_files(self.cache_folder)

    @staticmethod
    def clear_cache_files(cache_folder):
        if cache_folder is not None:
            if os.path.isdir(cache_folder):
                for the_file in os.listdir(cache_folder):
                    file_path = os.path.join(cache_folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            if not file_path.endswith("DND"):
                                shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

