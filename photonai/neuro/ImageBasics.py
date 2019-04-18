from ..photonlogger.Logger import Logger
from ..validation.ResultsDatabase import ParallelData
from sklearn.base import BaseEstimator
from skimage.util.shape import view_as_windows
from multiprocessing import Process, Queue, current_process, Value
from nilearn.image import resample_img, smooth_img
from nibabel.nifti1 import Nifti1Image
from pymodm.errors import DoesNotExist
from pymodm.files import File as PymodmFile
import queue
import time
import os
import pickle
import numpy as np
from pymongo import MongoClient
import gridfs



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
        self.client = MongoClient('trap-umbriel', 27017)
        self.db = self.client['photon_cache']
        self.fs = gridfs.GridFS(self.db)

    def fit(self, X, y):
        return self

    class ImageJob:

        def __init__(self, data, delegate, delegate_kwargs, saved_to_db: bool = False,
                     transform_name='transforming mri image'):
            self.data = data
            self.saved_to_db = saved_to_db
            self.delegate = delegate
            self.delegate_kwargs = delegate_kwargs
            self.transform_name = transform_name

    @staticmethod
    def parallel_application(folds_to_do, folds_done, num_jobs_done):

        client = MongoClient('trap-umbriel', 27017)
        db = client['photon_cache']
        fs = gridfs.GridFS(db)

        while True:
            try:
                task = folds_to_do.get_nowait()
            except queue.Empty:
                folds_done.close()
                break
            else:
                Logger().info(task.transform_name + " - " + str(os.getpid()))
                # print(task.transform_name + " - " + str(os.getpid()))
                # load data from db
                try:

                    # load data
                    if task.saved_to_db:
                        # get data obj
                        data_obj = ParallelData.objects.get({'_id': task.data})
                        data = pickle.loads(fs.get(data_obj.unprocessed_data).read())
                        if isinstance(data, np.ndarray):
                            if not data.flags['C_CONTIGUOUS']:
                                data = np.ascontiguousarray(data)
                    else:
                        data = task.data
                    # apply transform

                    delegate_output = task.delegate(data, **task.delegate_kwargs)
                    # binarize output
                    binary_output = pickle.dumps(delegate_output, protocol=2)
                    # save to db
                    data_obj.processed_data = fs.put(binary_output)
                    data_obj.save()
                except DoesNotExist as e:
                    Logger().error("Could not process task because data is not found with id " + str(task.data))
                # folds_done.put(fold_output)
                folds_done.put(task.data)
                num_jobs_done.value = num_jobs_done.value + 1
                # print(task.transform_name + " - " + str(os.getpid()) + " - DONE!")
                Logger().info(task.transform_name + " - " + str(os.getpid()) + " - DONE!")
        return True

    def apply_transform(self, X, delegate, transform_name="transformation", **transform_kwargs):

        output_images = []
        if self.nr_of_processes > 1:

            jobs_done = Queue()
            jobs_to_do = Queue()

            num_of_jobs_todo = 0
            for x_in in X:
                # print(hex(id(x_in)))
                # x_in = np.random.randint(0, 100)
                # save data to database
                data_entry = ParallelData()
                if isinstance(x_in, Nifti1Image):
                    x_in = x_in.dataobj
                fs_saved_data = self.fs.put(pickle.dumps(x_in, protocol=2))
                data_entry.unprocessed_data = fs_saved_data
                data_entry.save()
                # write new job to queue
                new_job = ImageTransformBase.ImageJob(data_entry._id, delegate, transform_kwargs,
                                                      saved_to_db=True,
                                                      transform_name=transform_name)
                num_of_jobs_todo += 1
                jobs_to_do.put(new_job)

            num_jobs_done = Value('i', 0)

            process_list = list()
            Logger().info("Nr of processes to create:" + str(self.nr_of_processes))
            for w in range(self.nr_of_processes):
                p = Process(target=ImageTransformBase.parallel_application, args=(jobs_to_do, jobs_done, num_jobs_done))
                process_list.append(p)
                p.start()

            # time.sleep(2)
            # print("collecting results")
            while num_jobs_done.value <= num_of_jobs_todo:
                while True:
                    try:
                        finished_data_id = jobs_done.get()
                        # get db entry
                        try:
                            loaded_data_obj = ParallelData.objects.get({'_id': finished_data_id})
                            processed_data = pickle.loads(self.fs.get(loaded_data_obj.processed_data).read())
                            if not self.output_img:
                                output_images.append(np.asarray(processed_data.dataobj))
                            else:
                                # print("appending img")
                                output_images.append(processed_data)
                            # after we loaded it, we can delete it in order to keep db nice and clean
                            self.fs.delete(loaded_data_obj.unprocessed_data)
                            self.fs.delete(loaded_data_obj.processed_data)
                            loaded_data_obj.delete()
                        except DoesNotExist:
                            Logger().error("Could not load processed data with id " + str(finished_data_id))
                    except queue.Empty:
                        # print("breaking queue because get resulting in empty error")
                        break
                    if num_jobs_done.value == num_of_jobs_todo and jobs_done.empty():
                        # print("breaking inner while because num of jobs is reached and jobs_done is empty")
                        break
                if num_jobs_done.value == num_of_jobs_todo and jobs_done.empty():
                    # print("breaking outer while because num of jobs is reached and jobs_done is empty")
                    break
                Logger().info("Waiting 2 seocnds before looking for data to collect")
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



class PatchImages(ImageTransformBase):

    def __init__(self, patch_size=25, random_state=42, nr_of_processes=3):
        Logger().info("Nr or processes: " + str(nr_of_processes))
        super(PatchImages, self).__init__(output_img=True, nr_of_processes=nr_of_processes)
        self.patch_size = patch_size
        self.random_state = random_state


    def transform(self, X, y=None, **kwargs):
        Logger().info("Drawing patches")
        transformed_X = self.apply_transform(X, PatchImages.draw_patches,
                                             transform_name="patching mri image",
                                             **{'patch_size': self.patch_size})

        return np.asarray(transformed_X)

    @staticmethod
    def draw_patches(patch_x, patch_size):

        Logger().info("drawing patch..!")

        if isinstance(patch_x, Nifti1Image):
            patch_x = patch_x.dataobj

        Benis = view_as_windows(patch_x, (patch_size, patch_size, 1), step=1)
        # print(Benis.shape)

        BenisLänge = Benis.shape[0]
        BenisBreite = Benis.shape[1]
        # BenisSchritte = BenisLänge / self.patch_size

        BenisMatrix = Benis[0:BenisLänge:patch_size, 0:BenisBreite:patch_size, :, :]
        # print(BenisMatrix.shape)

        # TODO: Reshape First 3 Matrix Dimensions into 1, which will give 900 images
        BenisMatrix = BenisMatrix.reshape((-1, BenisMatrix.shape[3], BenisMatrix.shape[4], BenisMatrix.shape[5]))
        BenisMatrix = np.squeeze(BenisMatrix)
        # print(BenisMatrix.shape)

        return BenisMatrix

    def copy_me(self):
        return PatchImages(self.patch_size, self.random_state, self.nr_of_processes)

    def _draw_single_patch(self):
        pass
