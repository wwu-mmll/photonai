from ..base.PhotonBase import PipelineBranch
from ..configuration.Register import PhotonRegister
from .ImageBasics import ImageTransformBase


class NeuroModuleBranch(PipelineBranch, ImageTransformBase):
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

    def __init__(self, name, nr_of_processes=1, cache_folder=None):
        super().__init__(name, nr_of_processes, cache_folder,
                         copy_delegate=True, output_img=False)

        self.has_hyperparameters = True
        self.needs_y = False
        self.needs_covariates = True
        self.current_config = None

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


    def transform(self, X, y=None, **kwargs):

        # build new copy
        # set params
        # set delegate function -> new copy.super-transform
        return self.apply_transform(X, )




    def set_params(self, **kwargs):
        self.current_config = kwargs
        super(NeuroModuleBranch, self).set_params(**kwargs)
