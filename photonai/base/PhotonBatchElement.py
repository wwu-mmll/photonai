from .PhotonBase import PipelineElement
from ..photonlogger.Logger import Logger
from .Helper import PHOTONDataHelper

import numpy as np


class PhotonBatchElement(PipelineElement):

    def __init__(self, name, hyperparameters: dict=None, test_disabled: bool=False, disabled: bool =False,
                 base_element=None, batch_size: int = 10, **kwargs):

        super(PhotonBatchElement, self).__init__(name, hyperparameters, test_disabled, disabled, base_element, **kwargs)
        # self.base_element = PipelineElement(base_element_name, hyperparameters=hyperparameters, **kwargs)

        self.batch_size = batch_size

    def transform(self, X, y=None, **kwargs):

        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            Logger().warn("Cannot do batching on a single entity.")
            return super(PhotonBatchElement, self).transform(X, y, **kwargs)

        # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        nr = PHOTONDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PHOTONDataHelper.chunker(nr, self.batch_size):

            batch_idx += 1
            Logger().debug(self.name + " is transforming batch nr " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PHOTONDataHelper.split_data(X, y, kwargs, start, stop)

            # call transform
            X_new, y_new, kwargs_new = self.adjusted_delegate_call(self.base_element.transform, X_batched, y_batched, **kwargs_dict_batched)

            # stack results
            processed_X, processed_y, processed_kwargs = PHOTONDataHelper.join_data(processed_X, X_new, processed_y, y_new,
                                                                                    processed_kwargs, kwargs_new)

        return processed_X, processed_y, processed_kwargs

    def predict(self, X, y=None, **kwargs):
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            Logger().warn("Cannot do batching on a single entity.")
            return super(PhotonBatchElement, self).predict(X, y, **kwargs)

        # initialize return values
        processed_y = None
        nr = PHOTONDataHelper.find_n(X)

        batch_idx = 0
        for start, stop in PHOTONDataHelper.chunker(nr, self.batch_size):

            batch_idx += 1
            Logger().debug(self.name + " is predicting batch nr " + str(batch_idx))

            # split data in batches
            X_batched, y_batched, kwargs_dict_batched = PHOTONDataHelper.split_data(X, y, kwargs, start, stop)

            # predict
            y_pred = super(PhotonBatchElement, self).predict(X_batched, y_batched, **kwargs_dict_batched)
            processed_y = PHOTONDataHelper.stack_results(y_pred, processed_y)

        return processed_y

    def copy_me(self):
        copy = PhotonBatchElement(name=self.name, hyperparameters=self.hyperparameters,
                                  batch_size=self.batch_size, **self.kwargs)

        if self.current_config is not None:
            copy.set_params(**self.current_config)
        return copy
