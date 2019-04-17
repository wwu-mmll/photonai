from sklearn.base import BaseEstimator
from photonai.photonlogger.Logger import Logger
from multiprocessing import Process, Queue, current_process, Value
import queue
import time
import os
from nilearn.image import resample_img, smooth_img
import numpy as np


# Smoothing and voxel-size Resampling capabilities
# img = True generates nilearn output images in memory;
# img = False generates np arrays for use in PHOTON Core
# PHOTON-Neuro internal format is always img=True
class ImageTransformBase(BaseEstimator):

    def __init__(self, output_img=True, nr_of_processes=1):
        self.output_img = output_img
        self.needs_y = False
        self.needs_covariates = False
        self.nr_of_processes = nr_of_processes

    def fit(self, X, y):
        return self

    class ImageJob:

        def __init__(self, data, delegate, delegate_kwargs, transform_name='transforming mri image'):
            self.data = data
            self.delegate = delegate
            self.delegate_kwargs = delegate_kwargs
            self.transform_name = transform_name

    @staticmethod
    def parallel_application(folds_to_do, folds_done, num_jobs_done):
        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                folds_done.close()
                break
            else:
                Logger().debug(task.transform_name + " - " + str(os.getpid()))
                print(task.transform_name + " - " + str(os.getpid()))
                fold_output = task.delegate(task.data, **task.delegate_kwargs)
                folds_done.put(fold_output)
                num_jobs_done.value = num_jobs_done.value + 1
                print(task.transform_name + " - " + str(os.getpid()) + " - DONE!")
        return True

    def apply_transform(self, X, delegate, transform_name="transformation", **transform_kwargs):

        output_images = []
        if self.nr_of_processes > 1:

            jobs_done = Queue()
            jobs_to_do = Queue()

            num_of_jobs_todo = 0
            for x_in in X:
                # print(hex(id(x_in)))
                new_job = ImageTransformBase.ImageJob(x_in, delegate, transform_kwargs, transform_name=transform_name)
                num_of_jobs_todo += 1
                jobs_to_do.put(new_job)

            num_jobs_done = Value('i', 0)

            process_list = list()
            for w in range(self.nr_of_processes):
                p = Process(target=ImageTransformBase.parallel_application, args=(jobs_to_do, jobs_done, num_jobs_done))
                process_list.append(p)
                p.start()

            # time.sleep(2)
            # print("collecting results")
            while num_jobs_done.value <= num_of_jobs_todo:
                while True:
                    try:
                        smoothed_img = jobs_done.get()
                        if not self.output_img:
                            output_images.append(np.asarray(smoothed_img.dataobj))
                        else:
                            # print("appending img")
                            output_images.append(smoothed_img)
                    except queue.Empty:
                        # print("breaking queue because get resulting in empty error")
                        break
                    if num_jobs_done.value == num_of_jobs_todo and jobs_done.empty():
                        # print("breaking inner while because num of jobs is reached and jobs_done is empty")
                        break
                if num_jobs_done.value == num_of_jobs_todo and jobs_done.empty():
                    # print("breaking outer while because num of jobs is reached and jobs_done is empty")
                    break
                time.sleep(2)
                # print(num_jobs_done.value)
                # print(num_jobs_done.value == num_of_jobs_todo)

            jobs_done.close()
            jobs_to_do.close()
            # print("finished collecting results")

            for p in process_list:
                # print("joining process " + str(p))
                p.join()
        else:
            output_images = delegate(X, **transform_kwargs)

        if not self.output_img:
            output_images = np.asarray(output_images)

        print("returning images: " + str(len(output_images)))
        return output_images


# Smoothing
class SmoothImages(ImageTransformBase):
    def __init__(self, fwhm=[2, 2, 2], output_img=True, nr_of_processes=3):
        super(SmoothImages, self).__init__(output_img, nr_of_processes)
        self.fwhm = fwhm

    def transform(self, X, y=None, **kwargs):
        kwargs_dict = {'fwhm': self.fwhm}
        return self.apply_transform(X, smooth_img, transform_name="smoothing mri image", **kwargs_dict)


class ResampleImages(ImageTransformBase):
    """
     Resampling voxel size
    """
    def __init__(self, voxel_size=[3, 3, 3], output_img=True, nr_of_processes=3):
        super(ResampleImages, self).__init__(output_img, nr_of_processes)
        self.voxel_size = voxel_size

    def transform(self, X, y=None, **kwargs):
        target_affine = np.diag(self.voxel_size)
        delegate_kwargs = {'target_affine': target_affine, 'interpolation': 'nearest'}
        return self.apply_transform(X, resample_img, transform_name="resampling mri image", **delegate_kwargs)

