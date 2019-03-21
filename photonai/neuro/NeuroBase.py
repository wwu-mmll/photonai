from ..base.PhotonBase import PipelineBranch, PipelineElement
from ..configuration.Register import PhotonRegister
from sklearn.base import BaseEstimator


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

    def __init__(self, base_element_name: str, hyperparameters: dict = {}, batch_size: int = 10, **kwargs):

        self.base_element = PipelineElement(base_element_name, hyperparameters=hyperparameters, **kwargs)
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
        return [(pos,pos + size) for pos in range(0, nr_items, size)]

    def batch_call(self, delegate, X, y=None, **kwargs):


        for X_new, y_new,

    def transform(self, X, y=None, **kwargs):



