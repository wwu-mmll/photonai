from ..photonlogger.Logger import Logger
from ..base.PhotonBatchElement import PhotonBatchElement
from sklearn.base import BaseEstimator
from skimage.util.shape import view_as_windows
from multiprocessing import Process, Queue, current_process, Value
from nilearn.image import resample_img, smooth_img
from nibabel.nifti1 import Nifti1Image
from pymodm.errors import DoesNotExist
import queue
import time
import os
import numpy as np
import json
import bcolz
import pickle
import shutil
import uuid


# Smoothing and voxel-size Resampling capabilities
# img = True generates nilearn output images in memory;
# img = False generates np arrays for use in PHOTON Core
# PHOTON-Neuro internal format is always img=True
class ImageTransformBase:

    def __init__(self, output_img=True, nr_of_processes=1, copy_delegate=False, batch_size=1, _cache_folder=None):
        self.output_img = output_img
        self.needs_y = False
        self.needs_covariates = False
        self.nr_of_processes = nr_of_processes
        self.copy_delegate = copy_delegate
        self.batch_size = batch_size
        self._cache_folder = _cache_folder

    @property
    def cache_folder(self):
        return self._cache_folder

    class ImageJob:

        def __init__(self, data, delegate, delegate_kwargs, saved_to_db: bool = False,
                     transform_name='transforming mri image', job_key=0, sort_index=0,
                     config_key=None):
            self.data = data
            self.saved_to_db = saved_to_db
            self.delegate = delegate
            self.delegate_kwargs = delegate_kwargs
            self.transform_name = transform_name
            self.sort_index = sort_index
            self.job_index = job_key
            self.config_key = config_key

    @staticmethod
    def parallel_application(folds_to_do, folds_done, num_jobs_done, cache_dir):

        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                folds_done.close()
                break
            else:
                try:

                    # load data
                    data = task.data
                    # apply transform
                    delegate_output = task.delegate(data, **task.delegate_kwargs)

                    # save ouput
                    save_filename = os.path.join(cache_dir, str(task.job_index))
                    if os.path.isdir(save_filename):
                        shutil.rmtree(save_filename)

                    with open(save_filename + ".p", 'wb') as f:
                        pickle.dump(delegate_output, f, protocol=2)

                except DoesNotExist as e:
                    print(e)

                folds_done.put((save_filename, task.sort_index))
                num_jobs_done.value = num_jobs_done.value + 1

        return True

    def apply_transform(self, X, delegate, transform_name="transformation", copy_object=None,
                        **transform_kwargs):
        """

        :param X: the data to which the delegate should be applied paralelly
        :param delegate: the function to call
        :param transform_name: how the function can be named in the logger
        :param config_dict: configuration of the whole pipeline used for caching
        :param copy_object: if the function should make a copy of the object instead of calling a function
        :param transform_kwargs: configuration to use in the delegate function
        :return:
        """

        if transform_kwargs is None:
            transform_kwargs = {}
        output_images = []
        if self.nr_of_processes > 1:

            jobs_done = Queue()
            jobs_to_do = Queue()
            num_jobs_done = Value('i', 0)

            # ----------- faking parallelization -----------------------
            # jobs_unparallel = list()
            # jobs_unparallel_done = list()
            # ----------- faking parallelization -----------------------

            num_of_jobs_todo = 0
            for start, stop in PhotonBatchElement.chunker(len(X), self.batch_size):

                # split data in batches
                # if dim > 1:
                #     x_in = X[start:stop, :]
                # else:
                x_in = X[start:stop]

                unique_key = uuid.uuid4()
                job_delegate = delegate
                if self.copy_delegate:
                    copy = copy_object.copy_me(with_parallelization_info=False)
                    job_delegate = getattr(copy, delegate)

                new_job = ImageTransformBase.ImageJob(data=x_in, delegate=job_delegate,
                                                      delegate_kwargs=transform_kwargs,
                                                      transform_name=transform_name,
                                                      job_key=unique_key,
                                                      sort_index=(start, stop))
                # job_cache_dict[config_dict_hash] = str(unique_key)
                jobs_to_do.put(new_job)

                # ----------- faking parallelization -----------------------
                # jobs_unparallel.append(new_job)
                # ----------- faking parallelization -----------------------

                num_of_jobs_todo += 1

            process_list = list()
            # Logger().info("Nr of processes to create:" + str(self.nr_of_processes))
            for w in range(self.nr_of_processes):
                p = Process(target=ImageTransformBase.parallel_application, args=(jobs_to_do, jobs_done, num_jobs_done,
                                                                                  self.cache_folder))
                process_list.append(p)
                p.start()

            # ----------- faking parallelization -----------------------
            # for job in jobs_unparallel:
            #     data = job.data
            #
            #     delegate_output = job.delegate(data, **job.delegate_kwargs)
            #
            #     save_filename = os.path.join(self.cache_folder, str(job.job_index))
            #     if os.path.isdir(save_filename):
            #         shutil.rmtree(save_filename)
            #
            #     with open(save_filename + ".p", 'wb') as f:
            #         pickle.dump(delegate_output, f, protocol=2)
            #     jobs_unparallel_done.append((save_filename, job.sort_index))

            # ----------- faking parallelization -----------------------

            sort_index_list =[]
            while len(output_images) < len(X):
                try:
                    (finished_data_id, sort_index) = jobs_done.get()
                    # ----------- faking parallelization -----------------------
                    # (finished_data_id, sort_index) = jobs_unparallel_done.pop()
                    # ----------- faking parallelization -----------------------

                    try:
                        # processed_data = bcolz.open(rootdir=finished_data_id)
                        with open(finished_data_id + ".p", 'rb') as of:
                            processed_data = pickle.load(of)
                        if not copy_object:
                            output_images.extend(processed_data)
                        else:
                            output_images.extend(processed_data[0])
                        sort_index_list.append(sort_index)
                    except DoesNotExist:
                        Logger().error("Could not load processed data with id " + str(finished_data_id))
                except queue.Empty:
                    pass

            jobs_done.close()
            jobs_to_do.close()
            # print("finished collecting results")
            # print("sorting results")
            sort_order = np.argsort([i[0] for i in sort_index_list])

            output_images_sorted = list()
            # Todo: list comprehension
            for idx in sort_order:
                start = sort_index_list[idx]
                for img in output_images[start[0]:start[1]]:
                    output_images_sorted.append(img)

            output_images = output_images_sorted

            if len(output_images) > 1:
                output_images = np.squeeze(np.asarray(output_images))

            for p in process_list:
                # print("joining process " + str(p))
                p.join()
        else:
            output_images = []
            if isinstance(X, str):
                X = [X]
            for el in X:
                output_images.append(delegate(el, **transform_kwargs))

        # if not self.output_img:
            # output_images = np.asarray(output_images)

        # if isinstance(output_images, list):
            # print("returning images: " + str(len(output_images)))
        return np.asarray(output_images)


# Smoothing
class SmoothImages(ImageTransformBase, BaseEstimator):
    def __init__(self, fwhm=[2, 2, 2], output_img=True, nr_of_processes=1):
        super(SmoothImages, self).__init__(output_img, nr_of_processes)

        # initialize private variable and
        self._fwhm = None
        self.fwhm = fwhm

    def fit(self, X, y=None, **kwargs):
        return self

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        if isinstance(fwhm, int):
            self._fwhm = [fwhm, fwhm, fwhm]
        elif isinstance(fwhm, list):
            if len(fwhm) != 3:
                raise Exception("fwhm parameter should be either an integer (3) or a in the form of [3, 3, 3]")
            else:
                self._fwhm = fwhm

    def transform(self, X, y=None, **kwargs):
        kwargs_dict = {'fwhm': self.fwhm}
        return self.apply_transform(X, smooth_img, transform_name="smoothing mri image", **kwargs_dict)


class ResampleImages(ImageTransformBase, BaseEstimator):
    """
     Resampling voxel size
    """
    def __init__(self, voxel_size=[3, 3, 3], output_img=True, nr_of_processes=1):
        super(ResampleImages, self).__init__(output_img, nr_of_processes)
        self._voxel_size = None
        self.voxel_size = voxel_size

    def fit(self, X, y=None, **kwargs):
        return self

    @property
    def voxel_size(self):
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size):
        if isinstance(voxel_size, int):
            self._voxel_size = [voxel_size, voxel_size, voxel_size]
        elif isinstance(voxel_size, list):
            if len(voxel_size) != 3:
                raise Exception("voxel_size parameter should be either an integer (3) or a in the form of [3, 3, 3]")
            else:
                self._voxel_size = voxel_size

    def transform(self, X, y=None, **kwargs):
        target_affine = np.diag(self.voxel_size)
        delegate_kwargs = {'target_affine': target_affine, 'interpolation': 'nearest'}
        return self.apply_transform(X, resample_img, transform_name="resampling mri image", **delegate_kwargs)


class PatchImages(ImageTransformBase, BaseEstimator):

    def __init__(self, patch_size=25, random_state=42, nr_of_processes=3):
        Logger().info("Nr or processes: " + str(nr_of_processes))
        super(PatchImages, self).__init__(output_img=True, nr_of_processes=nr_of_processes)
        # Todo: give cache folder to mother class

        self.patch_size = patch_size
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        Logger().info("Drawing patches")
        transformed_X = self.apply_transform(X, PatchImages.draw_patches,
                                             transform_name="patching mri image",
                                             **{'patch_size': self.patch_size})

        return transformed_X

    @staticmethod
    def draw_patches(patch_x, patch_size):
        if not isinstance(patch_x, list):
            return PatchImages.draw_patch_from_mri(patch_x, patch_size)
        else:
            return_list = []
            for p in patch_x:
                print(str(p))
                return_list.append(PatchImages.draw_patch_from_mri(p, patch_size))
            return return_list

    @staticmethod
    def draw_patch_from_mri(patch_x, patch_size):
        # Logger().info("drawing patch..")
        if isinstance(patch_x, str):
            from nilearn import image
            patch_x = np.ascontiguousarray(image.load_img(patch_x).get_data())

        if isinstance(patch_x, Nifti1Image):
            patch_x = np.ascontiguousarray(patch_x.dataobj)

        patches_drawn = view_as_windows(patch_x, (patch_size, patch_size, 1), step=1)

        patch_list_length = patches_drawn.shape[0]
        patch_list_width = patches_drawn.shape[1]

        output_matrix = patches_drawn[0:patch_list_length:patch_size, 0:patch_list_width:patch_size, :, :]

        # TODO: Reshape First 3 Matrix Dimensions into 1, which will give 900 images
        output_matrix = output_matrix.reshape((-1, output_matrix.shape[3], output_matrix.shape[4], output_matrix.shape[5]))
        output_matrix = np.squeeze(output_matrix)

        return output_matrix

    def copy_me(self):
        return PatchImages(self.patch_size, self.random_state, self.nr_of_processes)

    def _draw_single_patch(self):
        pass
