from collections.abc import Iterable

import numpy as np


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    from https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def __call__(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class PhotonPrintHelper:
    @staticmethod
    def _optimize_printing(pipe, config: dict):
        """
        make the sklearn config syntax prettily readable for humans
        """
        if pipe is None:
            return str(config)

        prettified_config = ["" + '\n']
        for el_key, el_value in config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in pipe.named_steps:
                new_pretty_key = '    ' + name + '->'
                prettified_config.append(new_pretty_key +
                                         pipe.named_steps[name].prettify_config_output(rest, el_value) + '\n')
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return ''.join(prettified_config)

    @staticmethod
    def config_to_human_readable_dict(pipe, specific_config):
        """
        """
        prettified_config = {}
        for el_key, el_value in specific_config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in pipe.named_steps:
                if not name in prettified_config:
                    prettified_config[name] = list()
                prettified_config[name].append(pipe.named_steps[name].prettify_config_output(rest, el_value))
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return prettified_config


class PhotonDataHelper:

    @staticmethod
    def chunker(nr_items, size):
        return [(pos, pos + size) for pos in range(0, nr_items, size)]

    @staticmethod
    def find_n(X):
        if hasattr(X, 'shape'):
            n = X.shape[0]
        elif isinstance(X, Iterable):
            n = len(X)
        else:
            n = 1
        return n

    @staticmethod
    def split_data(X, y, kwargs, start=0, stop=1, indices=None):
        # iterate through data batchwise

        # is that necessary?
        # if isinstance(X, np.ndarray):
        #     dim = len(X.shape)
        # else:
        #     dim = 1
        #
        # if dim > 1:
        #     X_batched = X[start:stop, :]
        # else:
        #     X_batched = X[start:stop]
        # ----------------------------------
        # if we want only one item, we need to make the stop value the excluded upper limit
        if start == stop:
            stop = start + 1

        if indices is None:
            indices_to_use = slice(start, stop)
        else:
            indices_to_use = indices

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X_batched = X[indices_to_use]

        # if we are to batch then apply it
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            y_batched = y[indices_to_use]
        else:
            y_batched = None

        kwargs_dict_batched = dict()
        if kwargs is not None:
            for key, kwargs_list in kwargs.items():
                if not isinstance(kwargs_list, np.ndarray):
                    kwargs_list = np.array(kwargs_list)
                if len(kwargs_list.shape) > 1:
                    kwargs_dict_batched[key] = kwargs_list[indices_to_use, :]
                else:
                    kwargs_dict_batched[key] = kwargs_list[indices_to_use]

        return X_batched, y_batched, kwargs_dict_batched

    @staticmethod
    def join_data(X, X_new, y, y_new, kwargs, kwargs_new):
        processed_X = PhotonDataHelper.stack_data_vertically(X, X_new)

        if y_new is not None:
            processed_y = PhotonDataHelper.stack_data_vertically(y, y_new)
        else:
            processed_y = None

        processed_kwargs = PhotonDataHelper.join_dictionaries(kwargs, kwargs_new)
        return processed_X, processed_y, processed_kwargs

    @staticmethod
    def join_dictionaries(dict_a: dict, dict_b: dict):
        new_dict = dict()

        if dict_a is None:
            dict_a = dict()
        if dict_b is not None:
            for key, value in dict_b.items():
                if key not in dict_a:
                    new_dict[key] = dict_b[key]
                else:
                    new_dict[key] = PhotonDataHelper.stack_data_vertically(dict_a[key], value)
        return new_dict

    @staticmethod
    def index_dict(d: dict, boolean_index):
        new_dict = dict()
        for key, value in d.items():
            new_dict[key] = value[boolean_index]
        return new_dict

    @staticmethod
    def stack_data_vertically(existing_array, new_array):
        if existing_array is not None and len(existing_array) != 0:
            if isinstance(new_array, np.ndarray) and len(new_array.shape) < 2:
                existing_array = np.hstack((existing_array, new_array))
            elif isinstance(new_array, list):
                existing_array = existing_array + new_array
            elif new_array is None and len(existing_array) == 0:
                return None
            else:
                existing_array = np.vstack((existing_array, new_array))
        else:
            existing_array = new_array
        return existing_array

    @staticmethod
    def stack_data_horizontally(existing_array, new_array):
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
        if existing_array is None or (isinstance(existing_array, np.ndarray) and existing_array.size == 0):
            existing_array = new_array
        else:
            # Todo: check for right dimensions!
            if existing_array.ndim == 1 and new_array.ndim == 1:
                existing_array = np.column_stack((existing_array, new_array))
            else:
                if new_array.ndim == 1:
                    new_array = np.reshape(new_array, (new_array.shape[0], 1))
                # a = np.concatenate((a, b), 1)
                existing_array = np.concatenate((existing_array, new_array), axis=1)
        return existing_array

    @staticmethod
    def resort_splitted_data(X, y, kwargs, idx_list):
        _sort_order = np.argsort(idx_list)
        X = np.asarray(X)[_sort_order]
        if y is not None:
            y = np.asarray(y)[_sort_order]
        if kwargs is not None:
            for k, v in kwargs.items():
                kwargs[k] = v[_sort_order]
        return X, y, kwargs

