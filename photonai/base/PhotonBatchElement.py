from .PhotonBase import PipelineElement
from ..photonlogger.Logger import Logger
import numpy as np


class PhotonBatchElement(PipelineElement):

    def __init__(self, name, hyperparameters: dict=None, test_disabled: bool=False, disabled: bool =False,
                 base_element=None, batch_size: int = 10, **kwargs):

        super(PhotonBatchElement, self).__init__(name, hyperparameters, test_disabled, disabled, base_element, **kwargs)
        # self.base_element = PipelineElement(base_element_name, hyperparameters=hyperparameters, **kwargs)

        self.batch_size = batch_size

    @staticmethod
    def chunker(nr_items, size):
        return [(pos, pos + size) for pos in range(0, nr_items, size)]

    def batch_call(self, delegate, X, y=None, call_with_y=True, **kwargs):

        # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        # iterate through data batchwise
        if isinstance(X, np.ndarray):
            nr = X.shape[0]
            dim = len(X.shape)
        else:
            nr = len(X)
            dim = 1

        batch_idx = 0
        for start, stop in PhotonBatchElement.chunker(nr, self.batch_size):

            batch_idx += 1
            Logger().debug(self.name + " is processing batch nr " + str(batch_idx))

            # split data in batches
            if dim > 1:
                X_batched = X[start:stop, :]
            else:
                X_batched = X[start:stop]

            # we are probably None anyway
            y_batched = y
            # if we are to batch then apply it
            if call_with_y and y is not None:
                y_batched = y[start:stop]

            kwargs_dict_batched = dict()
            for key, kwargs_list in kwargs.items():
                if not isinstance(kwargs_list, np.ndarray):
                    kwargs_list = np.array(kwargs_list)
                if len(kwargs_list.shape) > 1:
                    kwargs_dict_batched[key] = kwargs_list[start:stop, :]
                else:
                    kwargs_dict_batched[key] = kwargs_list[start:stop]

            # call the delegate
            X_new, y_new, kwargs_new = self.adjusted_delegate_call(delegate, X_batched, y_batched, **kwargs_dict_batched)

            # stack results
            processed_X = PhotonBatchElement.stack_results(X_new, processed_X)

            if call_with_y:
                processed_y = PhotonBatchElement.stack_results(y_new, processed_y)
                for proc_key, proc_values in kwargs_new.items():
                    new_kwargs_data = kwargs_new[proc_key]
                    if proc_key not in processed_kwargs:
                        processed_kwargs[proc_key] = new_kwargs_data
                    else:
                        processed_kwargs[proc_key] = PhotonBatchElement.stack_results(new_kwargs_data, processed_kwargs[proc_key])
            else:
                processed_kwargs = kwargs
                processed_y = y
        return processed_X, processed_y, processed_kwargs

    @staticmethod
    def stack_results(new_a, existing_a):
        if existing_a is not None:
            if isinstance(new_a, list) or (isinstance(new_a, np.ndarray) and len(new_a.shape) < 2):
                if isinstance(existing_a, list):
                    existing_a = existing_a + new_a
                else:
                    existing_a = np.hstack((existing_a, new_a))
            else:
                existing_a = np.vstack((existing_a, new_a))
        else:
            existing_a = new_a
        return existing_a

    def transform(self, X, y=None, **kwargs):
        return self.batch_call(self.base_element.transform, X, y, **kwargs)

    def predict(self, X, y=None, **kwargs):
        return self.batch_call(self.base_element.predict, X, y, call_with_y=False, **kwargs)

    def copy_me(self):
        copy = PhotonBatchElement(name=self.name, hyperparameters=self.hyperparameters,
                                  batch_size=self.batch_size, **self.kwargs)

        if self.current_config is not None:
            copy.set_params(**self.current_config)
        return copy
