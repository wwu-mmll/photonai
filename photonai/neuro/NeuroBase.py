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

    def test_transform(self, X, nr_of_tests=1, save_to_folder='.'):

        self.output_img = True
        nr_of_tested = 0

        filename = self.name + "_testcase_"

        for x_el in X:
            if nr_of_tested > nr_of_tests:
                break

            new_pic = self.transform(x_el)
            new_pic.to_filename(filename + str(nr_of_tested) + ".nii")

            nr_of_tested += 1
        self.output_img = False

    def transform(self, X, y=None, **kwargs):

        if self.nr_of_processes > 1:
            # build new copy
            # set params
            # set delegate function -> new copy.base_element.transform
            return self.apply_transform(X, delegate='base_element.transform',
                                        transform_name='applying neuro methods',
                                        copy_object=True)
        else:
            return self.base_element.transform(X, y, **kwargs)

    def set_params(self, **kwargs):
        self.current_config = kwargs
        super(NeuroModuleBranch, self).set_params(**kwargs)
