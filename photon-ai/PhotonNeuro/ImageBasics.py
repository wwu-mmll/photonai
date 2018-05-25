from sklearn.base import BaseEstimator
import numpy as np
from Logging.Logger import Logger

# Smoothing and voxel-size Resampling capabilities

# img = True generates nilearn output images in memory;
# img = False generates np arrays for use in PHOTON Core
# PHOTON-Neuro internal format is always img=True

# Smoothing
class SmoothImgs(BaseEstimator):
    def __init__(self, fwhr=[2, 2, 2], output_img=True):
        self.fwhr = fwhr
        self.output_img = output_img

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        Logger().info('Smoothing data with ' + str(self.fwhr))
        from nilearn.image import smooth_img
        out_imgs = []
        for x_in in X:
            smImg = smooth_img(x_in, self.fwhr)

            if not self.output_img:
                out_imgs.append(np.asarray(smImg.dataobj))
            else:
                out_imgs.append(smImg)

        if not self.output_img:
            Logger().info('Generating numpy array.')
            out_imgs = np.asarray(out_imgs)

        return out_imgs


# Resampling voxel size
class ResamplingImgs(BaseEstimator):
    def __init__(self, voxel_size=[3, 3, 3], output_img=True):
        self.voxel_size = voxel_size
        self.output_img = output_img

    def fit(self, X, y):
        self

    def transform(self, X, y=None):
        from nilearn.image import resample_img
        target_affine = np.diag(self.voxel_size)
        Logger().info('Resampling Voxel Size to ' + str(self.voxel_size))
        out_imgs = []
        for x_in in X:
            resImg = resample_img(x_in, target_affine=target_affine, interpolation='nearest')

            if not self.output_img:
                out_imgs.append(np.asarray(resImg.dataobj))
            else:
                out_imgs.append(resImg)

        if not self.output_img:
            Logger().info('Generating numpy array.')
            out_imgs = np.asarray(out_imgs)

        return out_imgs


#if __name__ == '__main__':
#
#     from nilearn.datasets import MNI152_FILE_PATH
#     dataset_files = [MNI152_FILE_PATH, MNI152_FILE_PATH]
#
#     resImg = ResamplingImgs(voxel_size=[5, 5, 5], output_img=True).transform(dataset_files)
#     smImg = SmoothImgs(fwhr=[6, 6, 6], output_img=True).transform(resImg)
#
#     from Photon_Neuro0.BrainAtlas import BrainAtlas
#     myAtlas = BrainAtlas(atlas_name='AAL',
#                          extract_mode='vec',
#                          whichROIs='all')
#     roi_data = myAtlas.transform(X=smImg)
#     myAtlas.getInfo()
#
#     Logger().info('')
