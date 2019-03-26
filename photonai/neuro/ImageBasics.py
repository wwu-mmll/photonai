from sklearn.base import BaseEstimator
from photonai.photonlogger.Logger import Logger
from multiprocessing import Pool, Process, Queue, current_process
import queue
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

        def __init__(self, data, delegate, delegate_kwargs):
            self.data = data
            self.delegate = delegate
            self.delegate_kwargs = delegate_kwargs

    @staticmethod
    def parallel_application(folds_to_do, folds_done):
        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                break
            else:
                fold_output = task.delegate(task.data, **task.delegate_kwargs)
                folds_done.put(fold_output)
        return True

    def apply_transform(self, X, delegate, transform_kwargs):

        jobs_done = Queue()
        jobs_to_do = Queue()

        for x_in in X:
            new_job = ImageTransformBase.ImageJob(x_in, delegate, transform_kwargs)
            jobs_to_do.put(new_job)

        process_list = list()
        for w in range(self.nr_of_processes):
            p = Process(target=ImageTransformBase.parallel_application, args=(jobs_to_do, jobs_done))
            process_list.append(p)
            p.start()

        for p in process_list:
            p.join()

        while not jobs_done.empty():
            output_images = []
            smoothed_img = self.folds_done.get()
            if not self.output_img:
                output_images.append(np.asarray(smoothed_img.dataobj))
            else:
                output_images.append(smoothed_img)

        if not self.output_img:
            output_images = np.asarray(output_images)

        return output_images


# Smoothing
class SmoothImages(ImageTransformBase):
    def __init__(self, fwhr=[2, 2, 2], output_img=True, nr_of_processes=3):
        super(SmoothImages, self).__init__(output_img, nr_of_processes)
        self.fwhr = fwhr

    def transform(self, X, y=None, **kwargs):
        return self.apply_transform(X, smooth_img, **{'fwhm': self.fwhr})


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
        return self.apply_transform(X, resample_img, **delegate_kwargs)
