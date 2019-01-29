from photonai.base.PhotonBase import PipelineBranch
from photonai.configuration.Register import PhotonRegister


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



