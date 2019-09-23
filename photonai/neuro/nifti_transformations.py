import numpy as np

from sklearn.base import BaseEstimator
from nilearn.image import resample_img, smooth_img, index_img
from nibabel.nifti1 import Nifti1Image

from photonai.photonlogger.logger import logger



class NeuroTransformerMixin:

    def __init__(self):
        self.output_img = False


# Smoothing
class SmoothImages(BaseEstimator, NeuroTransformerMixin):
    def __init__(self, fwhm=[2, 2, 2]):

        super(SmoothImages, self).__init__()

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

        if isinstance(X, list) and len(X) == 1:
            smoothed_img = smooth_img(X[0], fwhm=self.fwhm)
        elif isinstance(X, str):
            smoothed_img = smooth_img(X, fwhm=self.fwhm)
        else:
            smoothed_img = smooth_img(X, fwhm=self.fwhm)

        if not self.output_img:
            if isinstance(smoothed_img, list):
                smoothed_img = np.asarray([img.dataobj for img in smoothed_img])
            else:
                return smoothed_img.dataobj
        return smoothed_img


class ResampleImages(BaseEstimator, NeuroTransformerMixin):
    """
     Resampling voxel size
    """
    def __init__(self, voxel_size=[3, 3, 3]):
        super(ResampleImages, self).__init__()
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

        if isinstance(X, list) and len(X) == 1:
            resampled_img = resample_img(X[0], target_affine=target_affine, interpolation='nearest')
        elif isinstance(X, str):
            resampled_img = resample_img(X, target_affine=target_affine, interpolation='nearest')
        else:
            resampled_img = resample_img(X, target_affine=target_affine, interpolation='nearest')

        if self.output_img:
            if len(resampled_img.shape) == 3:
                if isinstance(resampled_img, (list, np.ndarray)):
                    return resampled_img
                else:
                    return [resampled_img]
            else:
                resampled_img = [index_img(resampled_img, i) for i in range(resampled_img.shape[-1])]
        else:
            if len(resampled_img.shape) == 3:
                return resampled_img.dataobj
            else:
                resampled_img = np.moveaxis(resampled_img.dataobj, -1, 0)

        return resampled_img


class PatchImages(BaseEstimator):

    def __init__(self, patch_size=25, random_state=42, nr_of_processes=3):
        logger.info("Nr or processes: " + str(nr_of_processes))
        super(PatchImages, self).__init__(output_img=True, nr_of_processes=nr_of_processes)
        # Todo: give cache folder to mother class

        self.patch_size = patch_size
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        logger.info("Drawing patches")
        return self.draw_patches(X, self.patch_size)


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
        # logger.info("drawing patch..")
        if isinstance(patch_x, str):
            from nilearn import image
            patch_x = np.ascontiguousarray(image.load_img(patch_x).get_data())

        if isinstance(patch_x, Nifti1Image):
            patch_x = np.ascontiguousarray(patch_x.dataobj)

        # Todo: import is failing; why?
        from skimage.util.shape import view_as_windows
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
