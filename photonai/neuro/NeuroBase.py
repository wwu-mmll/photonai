from ..base.PhotonBase import PipelineBranch
from ..configuration.Register import PhotonRegister
from ..base.PhotonBatchElement import PhotonBatchElement
from ..neuro.BrainAtlas import BrainAtlas
from nibabel.nifti1 import Nifti1Image
from multiprocessing import Process, Queue, current_process, Value
import numpy as np
import queue
import os
import uuid


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

    def __init__(self, name, nr_of_processes=1, output_img: bool = False):
        PipelineBranch.__init__(self, name)

        self.nr_of_processes = nr_of_processes
        self.output_img = output_img

        self.has_hyperparameters = True
        self.needs_y = False
        self.needs_covariates = True

        self.current_config = None

        self.skip_caching = True
        self.fix_fold_id = True
        self.do_not_delete_cache_folder = True

    def fit(self, X, y=None, **kwargs):
        # do nothing here!!
        return self

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

        if self.nr_of_processes > 1 and self.base_element.cache_folder is not None:
            # at first apply the transformation on several cores, everything gets written to the cache,
            # so the next step only has to reload the data ...
            self.apply_transform_parallelilzed(X)

        # do it item-wise for caching
        output = list()
        for x_in in X:
            x_new, _, _ = self.base_element.transform([x_in])
            output.append(x_new[0])
        output = np.asarray(output)
        return output, y, kwargs

    def set_params(self, **kwargs):
        self.current_config = kwargs
        super(NeuroModuleBranch, self).set_params(**kwargs)

    def copy_me(self):
        new_copy = super().copy_me()
        new_copy.base_element.cache_folder = self.base_element.cache_folder
        new_copy.nr_of_processes = self.nr_of_processes

        return new_copy

    class ImageJob:

        def __init__(self, data, delegate, job_key=0, sort_index=0):
            self.data = data
            self.delegate = delegate
            self.sort_index = sort_index
            self.job_index = job_key

    @staticmethod
    def parallel_application(folds_to_do):
        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                break
            else:

                # load data
                data = task.data
                # apply transform
                task.delegate(data, **task.delegate_kwargs)
        return True

    def apply_transform_parallelilzed(self, X):
        """

        :param X: the data to which the delegate should be applied paralelly
        """

        if self.nr_of_processes > 1:

            jobs_to_do = Queue()

            # ----------- faking parallelization -----------------------
            # jobs_unparallel = list()
            # ----------- faking parallelization -----------------------

            num_of_jobs_todo = 0
            for start, stop in PhotonBatchElement.chunker(len(X), self.batch_size):

                # split data in batches
                # if dim > 1:
                #     x_in = X[start:stop, :]
                # else:
                x_in = X[start:stop]
                unique_key = uuid.uuid4()

                # copy my pipeline
                new_pipe_copy = self.base_element.copy_me()
                new_pipe_copy.fix_fold_id = self.fix_fold_id
                new_pipe_copy.cache_folder = self.base_element.cache_folder
                new_pipe_copy.do_not_delete_cache_folder = self.do_not_delete_cache_folder
                new_pipe_copy.skip_loading = True

                job_delegate = new_pipe_copy.transform
                new_job = NeuroModuleBranch.ImageJob(data=x_in, delegate=job_delegate,
                                                     job_key=unique_key,
                                                     sort_index=(start, stop))

                jobs_to_do.put(new_job)

                # ----------- faking parallelization -----------------------
                # jobs_unparallel.append(new_job)
                # ----------- faking parallelization -----------------------

                num_of_jobs_todo += 1

            process_list = list()
            # Logger().info("Nr of processes to create:" + str(self.nr_of_processes))
            for w in range(self.nr_of_processes):
                p = Process(target=NeuroModuleBranch.parallel_application, args=(jobs_to_do))
                process_list.append(p)
                p.start()

            # ----------- faking parallelization -----------------------
            # for job in jobs_unparallel:
            #    job.delegate(job.data)
            # ----------- faking parallelization -----------------------

            for p in process_list:
                # print("joining process " + str(p))
                p.join()
