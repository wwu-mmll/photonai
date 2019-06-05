from ..photonlogger.Logger import Logger
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
class ImageTransformBase(BaseEstimator):

    def __init__(self, output_img=True, nr_of_processes=1, cache_folder=None):
        self.output_img = output_img
        self.needs_y = False
        self.needs_covariates = False
        self.nr_of_processes = nr_of_processes
        self.cache_folder = cache_folder
        # self.client = MongoClient('trap-umbriel', 27017)
        # self.db = self.client['photon_cache']
        # self.fs = gridfs.GridFS(self.db)

    def fit(self, X, y):
        return self

    def clear_cache(self):
        if self.cache_folder is not None:
            if os.path.isdir(self.cache_folder):
                shutil.rmtree(self.cache_folder)

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

        # fs = gridfs.GridFS(db)

        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                folds_done.close()
                break
            else:
                # Logger().info(task.transform_name + " - " + str(os.getpid()))
                # print(task.transform_name + " - " + str(os.getpid()))

                # load data from db
                try:

                    # load data
                    data = task.data
                    # apply transform
                    delegate_output = task.delegate(data, **task.delegate_kwargs)

                    # save ouput
                    save_filename = os.path.join(cache_dir, str(task.job_index))
                    if os.path.isdir(save_filename):
                        shutil.rmtree(save_filename)

                    # c = bcolz.carray(delegate_output, rootdir=save_filename)
                    # c.flush()
                    with open(save_filename + ".p", 'wb') as f:
                        pickle.dump(delegate_output, f, protocol=2)
                    # print("Process " + str(os.getpid()) + " finished job nr " + str(num_jobs_done.value))

                    # binarize output

                    # # save to db
                    # data_obj.processed_data = fs.put(binary_output)
                    # data_obj.save()
                except DoesNotExist as e:
                    print(e)
                    # Logger().error("Could not process task because data is not found with id " + str(task.data))
                # folds_done.put(fold_output)
                folds_done.put((save_filename, task.sort_index))
                num_jobs_done.value = num_jobs_done.value + 1
                # print(task.transform_name + " - " + str(os.getpid()) + " - DONE!")
                # Logger().info(task.transform_name + " - " + str(os.getpid()) + " - DONE!")
        return True

    def apply_transform(self, X, delegate, transform_name="transformation", config_dict=None, **transform_kwargs):

        output_images = []
        if self.nr_of_processes > 1:

            jobs_done = Queue()
            jobs_to_do = Queue()
            num_jobs_done = Value('i', 0)

            process_name = "debug_parallel_"
            job_cache_dict_filename = os.path.join(self.cache_folder, process_name + "cache.p")
            if not os.path.isdir(self.cache_folder):
                os.mkdir(self.cache_folder)
            if os.path.isfile(job_cache_dict_filename):
                with open(job_cache_dict_filename, 'rb') as ojd:
                    job_cache_dict = pickle.load(ojd)
            else:
                job_cache_dict = {}

            num_of_jobs_todo = 0
            for x_in in X:

                # write new job to queue
                if config_dict is not None:
                    config_dict_copy = dict(config_dict)
                    config_dict_copy["data_in_key"] = x_in
                    config_dict_hash = json.dumps(config_dict_copy, sort_keys=True)
                else:
                    config_dict_hash = json.dumps(x_in)

                # write anything to dictionary so that is easily indexable
                if not config_dict_hash in job_cache_dict:
                    unique_key = uuid.uuid4()
                    new_job = ImageTransformBase.ImageJob(data=x_in, delegate=delegate,
                                                          delegate_kwargs=transform_kwargs,
                                                          transform_name=transform_name,
                                                          job_key=unique_key,
                                                          sort_index=num_of_jobs_todo,
                                                          config_key=config_dict_hash)
                    job_cache_dict[config_dict_hash] = str(unique_key)
                    jobs_to_do.put(new_job)
                else:
                    num_jobs_done.value = num_jobs_done.value + 1
                    jobs_done.put((os.path.join(self.cache_folder, job_cache_dict[config_dict_hash]), num_of_jobs_todo))

                num_of_jobs_todo += 1

            with open(job_cache_dict_filename, 'wb') as jd:
                pickle.dump(job_cache_dict, jd)

            process_list = list()
            # Logger().info("Nr of processes to create:" + str(self.nr_of_processes))
            for w in range(self.nr_of_processes):
                p = Process(target=ImageTransformBase.parallel_application, args=(jobs_to_do, jobs_done, num_jobs_done,
                                                                                  self.cache_folder))
                process_list.append(p)
                p.start()

            sort_index_list =[]
            while len(output_images) < num_of_jobs_todo:
                try:
                    (finished_data_id, sort_index) = jobs_done.get()
                    # print("collecting image nr " + str(sort_index))
                    try:
                        # processed_data = bcolz.open(rootdir=finished_data_id)
                        with open(finished_data_id + ".p", 'rb') as of:
                            processed_data = pickle.load(of)

                        output_images.append(processed_data)
                        sort_index_list.append(sort_index)
                    except DoesNotExist:
                        Logger().error("Could not load processed data with id " + str(finished_data_id))
                except queue.Empty:
                    pass

            jobs_done.close()
            jobs_to_do.close()
            # print("finished collecting results")
            # print("sorting results")
            sort_order = np.argsort(sort_index_list)
            output_images = [output_images[i] for i in sort_order]

            for p in process_list:
                # print("joining process " + str(p))
                p.join()
        else:
            output_images = []
            for el in X:
                output_images.append(delegate(el, **transform_kwargs))

        # if not self.output_img:
            # output_images = np.asarray(output_images)

        # if isinstance(output_images, list):
            # print("returning images: " + str(len(output_images)))
        return output_images


# Smoothing
class SmoothImages(ImageTransformBase):
    def __init__(self, fwhm=[2, 2, 2], output_img=True, nr_of_processes=3, cache_folder=None):
        super(SmoothImages, self).__init__(output_img, nr_of_processes, cache_folder)
        self.fwhm = fwhm

    def transform(self, X, y=None, **kwargs):
        kwargs_dict = {'fwhm': self.fwhm}
        return self.apply_transform(X, smooth_img, transform_name="smoothing mri image", **kwargs_dict)


class ResampleImages(ImageTransformBase):
    """
     Resampling voxel size
    """
    def __init__(self, voxel_size=[3, 3, 3], output_img=True, nr_of_processes=3, cache_folder=None):
        super(ResampleImages, self).__init__(output_img, nr_of_processes, cache_folder)
        self.voxel_size = voxel_size

    def transform(self, X, y=None, **kwargs):
        target_affine = np.diag(self.voxel_size)
        delegate_kwargs = {'target_affine': target_affine, 'interpolation': 'nearest'}
        return self.apply_transform(X, resample_img, transform_name="resampling mri image", **delegate_kwargs)


class PatchImages(ImageTransformBase):

    def __init__(self, patch_size=25, random_state=42, nr_of_processes=3, cache_folder=None):
        Logger().info("Nr or processes: " + str(nr_of_processes))
        super(PatchImages, self).__init__(output_img=True, nr_of_processes=nr_of_processes, cache_folder=cache_folder)
        # Todo: give cache folder to mother class

        self.patch_size = patch_size
        self.random_state = random_state

    def transform(self, X, y=None, **kwargs):
        Logger().info("Drawing patches")
        transformed_X = self.apply_transform(X, PatchImages.draw_patches,
                                             transform_name="patching mri image",
                                             config_dict={'patch_size': self.patch_size},
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
