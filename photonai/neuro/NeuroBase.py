from ..base.PhotonBase import PipelineBranch
from ..configuration.Register import PhotonRegister
from ..neuro.ImageBasics import ImageTransformBase
from ..neuro.BrainAtlas import BrainAtlas
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os


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
        PipelineBranch.__init__(self, name)
        ImageTransformBase.__init__(self, nr_of_processes=nr_of_processes,
                                    cache_folder=cache_folder,
                                    copy_delegate=True,
                                    output_img=False)
        # super().__init__(name, nr_of_processes, cache_folder,
        #                  copy_delegate=True, output_img=False)

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
            # as the neuro branch is parallelized and processes several images subsequently on
            # different cores, we need to stop the children to process on several cores as well
            self.pipeline_elements.append(pipe_element)
            self._prepare_pipeline()
        else:
            raise ValueError('PipelineElement {} is not part of the Neuro module:'.format(pipe_element.name))

        return self

    def test_transform(self, X, nr_of_tests=1, save_to_folder='.', **kwargs):


        nr_of_tested = 0

        if kwargs and len(kwargs) > 0:
            self.set_params(**kwargs)

        copy_of_me = self.copy_me()
        copy_of_me.nr_of_processes = 1
        copy_of_me.output_img = True
        for p_element in copy_of_me.pipeline_elements:
            if isinstance(p_element.base_element, BrainAtlas):
                p_element.base_element.extract_mode = 'img'

        filename = self.name + "_testcase_"

        for x_el in X:
            if nr_of_tested > nr_of_tests:
                break

            new_pic, _, _ = copy_of_me.transform(x_el)

            if isinstance(new_pic, list):
                new_pic = new_pic[0]
            if not isinstance(new_pic, Nifti1Image):
                raise ValueError("last element of branch does not return a nifti image")

            new_filename = os.path.join(save_to_folder, filename + str(nr_of_tested) + "_transformed.nii")
            new_pic.to_filename(new_filename)

            nr_of_tested += 1

    def transform(self, X, y=None, **kwargs):

        if self.nr_of_processes > 1:
            # build new copy
            # set params
            # set delegate function -> new copy.base_element.transform
            return self.apply_transform(X, delegate='base_element.transform',
                                        transform_name='applying neuro methods',
                                        copy_object=self)
        else:
            return self.base_element.transform(X, y, **kwargs)

    def set_params(self, **kwargs):
        self.current_config = kwargs
        super(NeuroModuleBranch, self).set_params(**kwargs)
