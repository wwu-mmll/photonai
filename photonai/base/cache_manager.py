import os
import shutil
import numpy as np
import joblib
import glob
from dask.distributed import Lock

from photonai.photonlogger.logger import logger


class CacheManager:

    __LOCK_STR = 'photon_cache_manager'

    def __init__(self, _hash=None, cache_folder=None, parallel_use: bool = False,
                 single_subject_caching: bool =False):
        self._hash = _hash
        self.cache_folder = cache_folder

        self.pipe_order = None
        self.cache_index = None
        self.state = None

        self.parallel_use = parallel_use
        self.single_subject_caching = single_subject_caching

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
                     first_data_hash=None, first_data_str: str = ''):
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

        self.read_cache_index()

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

        cache_query = self.generate_cache_key(pipe_element_name)
        if cache_query in self.cache_index:
            if not self.single_subject_caching:
                logger.debug("Loading data from cache for " + pipe_element_name + ": "
                             + str(self.state.nr_items) + " items " + self.state.first_data_str
                             + " - " + str(self.state.config))
            filename = self.cache_index[cache_query]
            # lock = Lock(filename)
            # lock.acquire()
            with open(filename, 'rb') as f:
                (X, y, kwargs) = joblib.load(f)

            return X, y, kwargs
        return None

    def generate_cache_key(self, pipe_element_name):
        config_hash = self._find_config_for_element(pipe_element_name)
        cache_query = (pipe_element_name, self.hash, config_hash, self.state.nr_items, self.state.first_data_hash)
        return hash(cache_query)

    def check_cache(self, pipe_element_name):
        cache_query = self.generate_cache_key(pipe_element_name)

        if cache_query in self.cache_index:
            return True
        else:
            return False

    def save_data_to_cache(self, pipe_element_name, data):
        cache_query = self.generate_cache_key(pipe_element_name)
        filename = os.path.join(self.cache_folder, str(cache_query) + ".p")
        self.cache_index[cache_query] = filename
        if not self.single_subject_caching:
            logger.debug("Saving data to cache for " + pipe_element_name + ": " + str(self.state.nr_items) + " items "
                         + self.state.first_data_str + " - " + str(self.state.config))

        # write cached data to filesystem
        with open(filename, 'wb') as f:
            joblib.dump(data, f)

    def read_cache_index(self):
        cached_files = glob.glob(os.path.join(self.cache_folder, "*.p"))
        self.cache_index = {int(os.path.splitext(os.path.basename(i))[0]): i for i in cached_files}

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
