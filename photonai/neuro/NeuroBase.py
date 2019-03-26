from ..base.PhotonBase import PipelineBranch, PipelineElement
from ..configuration.Register import PhotonRegister
from sklearn.base import BaseEstimator
import numpy as np


class NeuroModuleBranch(PipelineBranch):
    """
    A substream of neuro elements that are encapsulated into a single block of PipelineElements that all perform
    transformations on MRI data. A NeuroModuleBranch takes niftis or nifti paths as input and should pass a numpy array
    to the subsequent PipelineElements.

    Parameters
    ----------
    * `name` [str]:
        Name of the NeuroModule pipeline branch

    """
    NEURO_ELEMENTS = PhotonRegister.get_package_info(['PhotonNeuro'])

    def __init__(self, name):
        super().__init__(name)
        self.has_hyperparameters = True
        self.needs_y = False
        self.needs_covariates = True

    def __iadd__(self, pipe_element):
        """
        Add an element to the neuro branch. Only neuro pipeline elements are allowed.
        Returns self

        Parameters
        ----------
        * `pipe_element` [PipelineElement]:
            The transformer object to add. Should be registered in the Neuro module.
        """
        if pipe_element.name in NeuroModuleBranch.NEURO_ELEMENTS:
            self.pipeline_elements.append(pipe_element)
            self._prepare_pipeline()
        else:
            raise ValueError('PipelineElement {} is not part of the Neuro module:'.format(pipe_element.name))

        return self


class NeuroBatch(BaseEstimator):

    def __init__(self, base_element: object, batch_size: int = 10):

        # self.base_element = PipelineElement(base_element_name, hyperparameters=hyperparameters, **kwargs)
        self.base_element = base_element
        self.batch_size = batch_size

    def fit(self, X, y=None, **kwargs):
        self.base_element.fit(X, y, **kwargs)
        return self

    def set_params(self, **params):
        self.base_element.set_params(**params)

    def get_params(self, deep=True):
        return self.base_element.get_params(deep=deep)

    @staticmethod
    def chunker(nr_items, size):
        return [(pos, pos + size) for pos in range(0, nr_items, size)]

    def batch_call(self, delegate, X, y=None, call_with_y=True, **kwargs, ):

        # initialize return values
        processed_X = None
        processed_y = None
        processed_kwargs = dict()

        # iterate through data batchwise
        for start, stop in NeuroBatch.chunker(X.shape[0], self.batch_size):

            # split data in batches
            X_batched = X[start:stop, :]
            if call_with_y:
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
            if call_with_y:
                X_new, y_new, kwargs_new = delegate(X_batched, y_batched, **kwargs_dict_batched)
            else:
                X_new = delegate(X_batched, **kwargs_dict_batched)

            # stack results
            if processed_X is not None:
                processed_X = np.vstack((processed_X, X_new))
            else:
                processed_X = X_new

            if call_with_y:
                if processed_y is None:
                    processed_y = y_new
                else:
                    if len(processed_y.shape) > 1:
                        processed_y = np.vstack((processed_y, y_new))
                    else:
                        processed_y = np.hstack((processed_y, y_new))

                for proc_key, proc_values in kwargs_new.items():
                    new_kwargs_data = kwargs_new[proc_key]
                    if not isinstance(new_kwargs_data, np.ndarray):
                        new_kwargs_data = np.array(new_kwargs_data)
                    if not proc_key in processed_kwargs:
                        processed_kwargs[proc_key] = new_kwargs_data
                    else:
                        if len(new_kwargs_data.shape) > 1:
                            processed_kwargs[proc_key] = np.vstack((processed_kwargs[proc_key], new_kwargs_data))
                        else:
                            processed_kwargs[proc_key] = np.hstack((processed_kwargs[proc_key], new_kwargs_data))
            else:
                processed_kwargs = kwargs
                processed_y = y
        return processed_X, processed_y, processed_kwargs

    def transform(self, X, y=None, **kwargs):
        return self.batch_call(self.base_element.transform, X, y, **kwargs)

    def predict(self, X, y=None, **kwargs):
        return self.batch_call(self.base_element.predict, X, y, call_with_y=False, **kwargs)




