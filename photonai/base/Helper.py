from collections.abc import Iterable
import numpy as np

class PHOTONPrintHelper:
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
                prettified_config[name] = pipe.named_steps[name].prettify_config_output(rest, el_value)
            else:
                raise ValueError('Item is not contained in pipeline:' + name)
        return prettified_config


class PHOTONDataHelper:

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
    def split_data(X, y, kwargs, start, stop):
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

        X_batched = X[start:stop]

        # if we are to batch then apply it
        if y is not None:
            y_batched = y[start:stop]
        else:
            y_batched = None

        kwargs_dict_batched = dict()
        if kwargs is not None:
            for key, kwargs_list in kwargs.items():
                if not isinstance(kwargs_list, np.ndarray):
                    kwargs_list = np.array(kwargs_list)
                if len(kwargs_list.shape) > 1:
                    kwargs_dict_batched[key] = kwargs_list[start:stop, :]
                else:
                    kwargs_dict_batched[key] = kwargs_list[start:stop]
        return X_batched, y_batched, kwargs_dict_batched

    @staticmethod
    def join_data(X, X_new, y, y_new, kwargs, kwargs_new):
        processed_X = PHOTONDataHelper.stack_results(X_new, X)

        if y_new is not None:
            processed_y = PHOTONDataHelper.stack_results(y_new, y)
        else:
            processed_y = None

        processed_kwargs = PHOTONDataHelper.join_dictionaries(kwargs, kwargs_new)
        return processed_X, processed_y, processed_kwargs

    @staticmethod
    def join_dictionaries(kwargs, kwargs_new):
        processed_kwargs = kwargs
        if kwargs is None:
            kwargs = dict()
        if kwargs_new is not None and len(kwargs_new.items()) > 0:
            for proc_key, proc_values in kwargs_new.items():
                new_kwargs_data = kwargs_new[proc_key]
                if proc_key not in processed_kwargs:
                    processed_kwargs[proc_key] = new_kwargs_data
                else:
                    processed_kwargs[proc_key] = PHOTONDataHelper.stack_results(new_kwargs_data, kwargs[proc_key])
        return processed_kwargs

    @staticmethod
    def stack_results(new_a, existing_a):
        if existing_a is not None and len(existing_a) != 0:
            if isinstance(new_a, np.ndarray) and len(new_a.shape) < 2:
                existing_a = np.hstack((existing_a, new_a))
            elif isinstance(new_a, list):
                    existing_a = existing_a + new_a
            elif new_a is None and len(existing_a) == 0:
                return None
            else:
                existing_a = np.vstack((existing_a, new_a))
        else:
            existing_a = new_a
        return existing_a

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

