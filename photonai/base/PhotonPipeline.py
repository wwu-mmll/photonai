from sklearn.utils.metaestimators import _BaseComposition
import numpy as np
import datetime
import os
import pickle
import uuid
import shutil
from ..photonlogger.Logger import Logger
from ..base.Helper import PHOTONDataHelper


class PhotonPipeline(_BaseComposition):

    def __init__(self, steps):
        self.steps = steps

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
            self.cache_man = CacheManager(self._fold_id, self.cache_folder)

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
                os.mkdir(self._cache_folder)
            self.cache_man = CacheManager(self._fold_id, self.cache_folder)
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
                if self.caching and self.current_config is not None:
                    X, y, kwargs = self.load_or_save_cached_data(self._final_estimator.name, X, y, kwargs, self._final_estimator)
                else:
                    X, y, kwargs = self._final_estimator.transform(X, y, **kwargs)

        if self.caching:
            self.cache_man.save_cache_index()

        return X, y, kwargs

    def load_or_save_cached_data(self, name, X, y, kwargs, transformer, fit=False, needed_for_further_computation=False):
        if not self.single_subject_caching:
            # if we do it group-wise then its easy
            if self.skip_loading and not needed_for_further_computation:
                # check if data is already calculated
                if self.cache_man.check_cache(name):
                    # if so, do nothing
                    return X, y, kwargs
                else:
                    # otherwise, do the calculation and save it
                    cached_result = None
            else:
                start_time_for_loading = datetime.datetime.now()
                cached_result = self.cache_man.load_cached_data(name)

            if cached_result is None:
                X, y, kwargs = self._do_timed_fit_transform(name, transformer, fit, X, y, **kwargs)
                self.cache_man.save_data_to_cache(name, (X, y, kwargs))
            else:
                X, y, kwargs = cached_result[0], cached_result[1], cached_result[2]
                loading_duration = (datetime.datetime.now() - start_time_for_loading).total_seconds()
                n = PHOTONDataHelper.find_n(X)
                self.time_monitor['transform_cached'].append((name, loading_duration, n))
            return X, y, kwargs
        else:
            # if we do it subject-wise we need to iterate and collect the results
            processed_X, processed_y, processed_kwargs = list(), list(), dict()
            X_uncached, y_uncached, kwargs_uncached = list(), list(), dict()
            list_of_idx_cached, list_of_idx_non_cached = list(), list()

            nr = PHOTONDataHelper.find_n(X)
            for start, stop in PHOTONDataHelper.chunker(nr, 1):
                # split data in single entities
                X_batched, y_batched, kwargs_dict_batched = PHOTONDataHelper.split_data(X, y, kwargs, start, stop)
                self.cache_man.update_single_subject_state_info(X_batched)

                # check if item has been processed
                if self.cache_man.check_cache(name):
                    list_of_idx_cached.append(start)
                else:
                    list_of_idx_non_cached.append(start)
                    X_uncached.append(X_batched)
                    y_uncached.append(y_batched)
                    kwargs_uncached = PHOTONDataHelper.join_dictionaries(kwargs_uncached, kwargs_dict_batched)

            # now we know which part can be loaded and which part should be transformed
            # first apply the transformation to the group, then save it single-subject-wise
            if len(list_of_idx_non_cached) > 0:
                # apply transformation groupwise
                new_group_X, new_group_y, new_group_kwargs = self._do_timed_fit_transform(name, transformer, fit,
                                                                                          X_uncached,
                                                                                          y_uncached,
                                                                                          **kwargs_uncached)

                # then save it single
                for start, stop in PHOTONDataHelper.chunker(nr, 1):
                    # split data in single entities
                    X_batched, y_batched, kwargs_dict_batched = PHOTONDataHelper.split_data(new_group_X,
                                                                                            new_group_y,
                                                                                            new_group_kwargs,
                                                                                            start, stop)
                    self.cache_man.update_single_subject_state_info(X_batched)
                    self.cache_man.save_data_to_cache(name, (X_batched, y_batched, kwargs_dict_batched))

                # we need to collect the data only when we want to load them
                # we can skip that process if we only want them to get into the cache (case: parallelisation)
                if not self.skip_loading or needed_for_further_computation:
                    # stack results
                    processed_X, processed_y, processed_kwargs = new_group_X, new_group_y, new_group_kwargs

            # afterwards load everything that has been cached
            if len(list_of_idx_cached) > 0:
                if not self.skip_loading or needed_for_further_computation:
                    for cache_idx in list_of_idx_cached:
                        X_batched, y_batched, kwargs_dict_batched = PHOTONDataHelper.split_data(X, y,
                                                                                                kwargs, cache_idx,
                                                                                                cache_idx)
                        self.cache_man.update_single_subject_state_info(X_batched)

                        # time the loading of the cached item
                        start_time_for_loading = datetime.datetime.now()
                        transformed_X, transformed_y, transformed_kwargs = self.cache_man.load_cached_data(name)
                        loading_duration = (datetime.datetime.now() - start_time_for_loading).total_seconds()
                        self.time_monitor['transform_cached'].append((name, loading_duration, PHOTONDataHelper.find_n(X)))

                        processed_X, processed_y, processed_kwargs = PHOTONDataHelper.join_data(processed_X, transformed_X,
                                                                                                processed_y, transformed_y,
                                                                                                processed_kwargs, transformed_kwargs)
            # now sort the data in the correct order again
            processed_X, processed_y, processed_kwargs = PHOTONDataHelper.resort_splitted_data(processed_X,
                                                                                               processed_y,
                                                                                               processed_kwargs,
                                                                                               PHOTONDataHelper.stack_results(list_of_idx_non_cached,
                                                                                                                              list_of_idx_cached))

            return processed_X, processed_y, processed_kwargs

    def _do_timed_fit_transform(self, name, transformer, fit, X, y, **kwargs):

        n = PHOTONDataHelper.find_n(X)

        if fit:
            fit_start_time = datetime.datetime.now()
            transformer.fit(X, y, **kwargs)
            fit_duration = (datetime.datetime.now() - fit_start_time).total_seconds()
            self.time_monitor['fit'].append((name, fit_duration, n))

        transform_start_time = datetime.datetime.now()
        X, y, kwargs = transformer.transform(X, y, **kwargs)
        transform_duration = (datetime.datetime.now() - transform_start_time).total_seconds()
        self.time_monitor['transform_computed'].append((name, transform_duration, n))
        return X, y, kwargs

    def _caching_fit_transform(self, X, y, kwargs, fit=False):

        if self.caching:
            # update infos, just in case
            self.cache_man.hash = self._fold_id
            self.cache_man.cache_folder = self.cache_folder
            if not self.single_subject_caching:
                self.cache_man.prepare([name for name, e in self.steps], self.current_config, X)
            else:
                self.cache_man.prepare([name for name, e in self.steps], self.current_config, single_subject_caching=True)
            last_cached_item = None

        # all steps except the last one
        num_steps = len(self.steps) - 1

        for num, (name, transformer) in enumerate(self.steps[:-1]):
            if not self.caching or self.current_config is None or \
                    (hasattr(transformer, 'skip_caching') and transformer.skip_caching):
                X, y, kwargs = self._do_timed_fit_transform(name, transformer, fit, X, y, **kwargs)
            else:
                # load data when the first item occurs that needs new calculation
                if self.cache_man.check_cache(name):
                    # as long as we find something cached, we remember what it was
                    last_cached_item = name
                    # if it is the last step, we need to load the data now
                    if num + 1 == num_steps and not self.skip_loading:
                        X, y, kwargs = self.load_or_save_cached_data(last_cached_item, X, y, kwargs, transformer, fit)
                else:
                    if last_cached_item is not None:
                        # we load the cached data when the first transformation on this data is upcoming
                        X, y, kwargs = self.load_or_save_cached_data(last_cached_item, X, y, kwargs, transformer, fit,
                                                                     needed_for_further_computation=True)
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
                predict_start_time = datetime.datetime.now()
                y_pred = self._final_estimator.predict(X, **kwargs)
                predict_duration = (datetime.datetime.now() - predict_start_time).total_seconds()
                n = PHOTONDataHelper.find_n(X)
                self.time_monitor['predict'].append((self.steps[-1][0], predict_duration, n))
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

    def clear_cache(self):
        if self.cache_man is not None:
            self.cache_man.clear_cache()


class CacheManager:

    def __init__(self, _hash=None, cache_folder=None):
        self._hash = _hash
        self.cache_folder = cache_folder

        self.pipe_order = None
        self.cache_index = None
        self.state = None
        self.cache_file_name = None
        self.lock = None

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, value):
        if not isinstance(value, str):
            self._hash = str(value)
        else:
            self._hash = value

    def set_lock(self, lock):
        self.lock = lock

    class State:
        def __init__(self, config=None, nr_items=None,
                     first_data_hash=None, first_data_str: str = None):
            self.config = config
            self.nr_items = nr_items
            self.first_data_hash = first_data_hash
            self.first_data_str = first_data_str

    def update_single_subject_state_info(self, X):
        self.state.first_data_hash = hash(str(X[0]))
        if isinstance(X[0], str):
            self.state.first_data_str = X[0]
        else:
            self.state.first_data_str = str(self.state.first_data_hash)

    def prepare(self, pipe_elements, config, X=None, single_subject_caching=False):

        cache_name = 'photon_cache_index.p'
        self.cache_file_name = os.path.join(self.cache_folder, cache_name)

        if self.lock is not None:
            with self.lock.read_lock():
                self._read_cache_index()
        else:
            self._read_cache_index()

        self.pipe_order = pipe_elements

        self.state = CacheManager.State(config=config)

        if X is not None:
            self.state.first_data_hash = hash(str(X[0]))
            if isinstance(X, np.ndarray):
                self.state.nr_items = X.shape[0]
            else:
                self.state.nr_items = len(X)
            self.state.first_data_str = str(self.state.first_data_hash)

        if single_subject_caching:
            self.state.nr_items = 1

    def _read_cache_index(self):

        if os.path.isfile(self.cache_file_name):
            with open(self.cache_file_name, 'rb') as f:
                # print("Reading cache index ")
                try:
                    self.cache_index = pickle.load(f)
                    # print(len(self.cache_index))
                except EOFError as e:
                    print("EOF Error... retrying!")
                    print("Cache index loaded: " + str(self.cache_index))
                    # retry...
                    self._read_cache_index()
        else:
            self.cache_index = {}

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
                    if isinstance(key_value, list):
                        key_value = frozenset(key_value)
                    relevant_dict[key_name] = key_value

        return hash(frozenset(relevant_dict.items()))

    def load_cached_data(self, pipe_element_name):

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
        if self.lock is not None:
            with self.lock.write_lock():
                self._write_cache_index()
        else:
            self._write_cache_index()

    def _write_cache_index(self):
        with open(self.cache_file_name, 'wb') as f:
            # print("Writing cache index")
            # print(self.cache_index)
            pickle.dump(self.cache_index, f)

    def clear_cache(self):
        CacheManager.clear_cache_files(self.cache_folder)

    @staticmethod
    def clear_cache_files(cache_folder, force_all=False):
        if cache_folder is not None:
            if os.path.isdir(cache_folder):
                for the_file in os.listdir(cache_folder):
                    file_path = os.path.join(cache_folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            if not file_path.endswith("DND") or force_all:
                                shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

