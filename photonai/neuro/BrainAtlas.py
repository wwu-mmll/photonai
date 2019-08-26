import numpy as np
import os.path
import inspect
import glob
from nilearn.input_data import NiftiMasker
from nilearn import image, masking, _utils
from nilearn.masking import unmask
from nilearn._utils.niimg import _safe_get_data
import nibabel as nib
import time
from pathlib import Path
from sklearn.base import BaseEstimator

from ..photonlogger.Logger import Logger, Singleton


class RoiObject:

    def __init__(self, index=0, label='', size=0, mask=None):
        self.index = index
        self.label = label
        self.size = size
        self.mask = None
        self.is_empty = False


class MaskObject:

    def __init__(self, name: str = '', mask_file: str = '', mask = None):
        self.name = name
        self.mask_file = mask_file
        self.mask = None
        self.is_empty = False


class AtlasObject:

    def __init__(self, name='', path='', labels_file='', mask_threshold=None):
        self.name = name
        self.path = path
        self.labels_file = labels_file
        self.mask_threshold = mask_threshold
        self.indices = list()
        self.roi_list = list()
        self.map = None


@Singleton
class MaskLibrary:
    MASK_DICTIONARY = {'MNI_ICBM152_GrayMatter': 'mni_icbm152_gm_tal_nlin_sym_09a.nii.gz',
                       'MNI_ICBM152_WhiteMatter': 'mni_icbm152_wm_tal_nlin_sym_09a.nii.gz',
                       'MNI_ICBM152_WholeBrain': 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz',
                       'Cerebellum': 'P_08_Cere.nii.gz'}

    def __init__(self):
        self.mask_dir = os.path.dirname(inspect.getfile(BrainAtlas)) + '/' + 'Atlases/'
        self.available_masks = dict()
        self._inspect_mask_dir()
        self.loaded_masks = dict()

    def _inspect_mask_dir(self):
        for mask_id, mask_info in self.MASK_DICTIONARY.items():
            mask_file = glob.glob(os.path.join(self.mask_dir, '*/' + mask_info))[0]
            self.available_masks[mask_id] = MaskObject(name=mask_id,
                                                  mask_file=mask_file)
        return

    def _add_custom_mask(self, mask_file):
        if not os.path.isfile(mask_file):
            raise FileNotFoundError("Cannot find custom mask {}".format(mask_file))
        return MaskObject(name=os.path.basename(mask_file), mask_file=mask_file)

    def _load_mask(self, mask_name: str='', target_affine=None, target_shape=None):
        Logger().debug('Loading Mask')

        if mask_name in self.available_masks.keys():
            mask = self.available_masks[mask_name]
        else:
            Logger().debug("Checking custom mask")
            mask = self._add_custom_mask(mask_name)

        img = image.load_img(mask.mask_file)
        mask.mask = image.new_img_like(mask.mask_file, img.get_data() > 0)

        if target_affine is not None and target_shape is not None:
            mask.mask = image.resample_img(mask.mask, target_affine=target_affine, target_shape=target_shape,
                                           interpolation='nearest')

            # check orientations
            orient_data = ''.join(nib.aff2axcodes(target_affine))
            orient_roi = ''.join(nib.aff2axcodes(mask.mask.affine))
            if not orient_roi == orient_data:
                Logger().info('Orientation of mask and data are not the same: '
                              + orient_roi + ' (mask) vs. ' + orient_data + ' (data)')

        # check if roi is empty
        if np.sum(mask.mask.dataobj != 0) == 0:
            Logger().error('No voxels in mask after resampling (' + mask.name + ').')
            mask.is_empty = True

        self.loaded_masks[(mask.name, str(target_affine), str(target_shape))] = mask
        print("Loading mask done!")

    def get_mask(self, mask_name, target_affine=None, target_shape=None):

        if self.loaded_masks is None or (mask_name, str(target_affine), str(target_shape)) not in self.loaded_masks:
            Logger().debug("Loading mask from filesystem")
            self._load_mask(mask_name, target_affine, target_shape)

        return self.loaded_masks[(mask_name, str(target_affine), str(target_shape))]


@Singleton
class AtlasLibrary:
    ATLAS_DICTIONARY = {'AAL': 'AAL.nii.gz',
                        'HarvardOxford_Cortical_Threshold_25': 'HarvardOxford-cort-maxprob-thr25.nii.gz',
                        'HarvardOxford_Subcortical_Threshold_25': 'HarvardOxford-sub-maxprob-thr25.nii.gz',
                        'HarvardOxford_Cortical_Threshold_50': 'HarvardOxford-cort-maxprob-thr50.nii.gz',
                        'HarvardOxford_Subcortical_Threshold_50': 'HarvardOxford-sub-maxprob-thr50.nii.gz',
                        'Yeo_7': 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz',
                        'Yeo_7_Liberal': 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
                        'Yeo_17': 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz',
                        'Yeo_17_Liberal': 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
                        }

    def __init__(self):
        self.atlas_dir = os.path.dirname(inspect.getfile(BrainAtlas)) + '/' + 'Atlases/'
        self.available_atlasses = self._inspect_atlas_dir()
        self.loaded_atlasses = None

    def _inspect_atlas_dir(self):
        available_atlasses = dict()
        for atlas_id, atlas_info in self.ATLAS_DICTIONARY.items():
            atlas_file = glob.glob(os.path.join(self.atlas_dir, '*/' + atlas_info))[0]
            atlas_basename = os.path.basename(atlas_file)[:-7]
            atlas_dir = os.path.dirname(atlas_file)
            available_atlasses[atlas_id] = AtlasObject(name=atlas_id,
                                                       path=atlas_file,
                                                       labels_file=os.path.join(atlas_dir, atlas_basename + '_labels.txt'))
        return available_atlasses

    def list_rois(self, atlas: str):
        if atlas not in self.ATLAS_DICTIONARY.keys():
            Logger().info('Atlas {} is not supported.'.format(atlas))
            return

        atlas = self.get_atlas(atlas)
        roi_names = [roi.label for roi in atlas.roi_list]
        Logger().info(roi_names)
        return roi_names

    def _load_atlas(self, atlas_name, target_affine=None, target_shape=None, mask_threshold=None):

        print("Loading Atlas")
        # load atlas object and copy it
        atlas_obj_orig = self.available_atlasses[atlas_name]
        atlas_obj = AtlasObject(atlas_obj_orig.name, atlas_obj_orig.path,
                                atlas_obj_orig.labels_file,
                                atlas_obj_orig.mask_threshold)
        atlas_obj.map = atlas_obj_orig.map
        atlas_obj.indices = atlas_obj_orig.indices
        atlas_obj.map = image.load_img(atlas_obj.path).get_data()

        # load it and adapt it to target affine and target shape
        if mask_threshold is not None:
            atlas_obj.map = atlas_obj.map > self.mask_threshold  # get actual map data with probability maps
            atlas_obj.map = atlas_obj.map.astype(int)

        atlas_obj.indices = list(np.unique(atlas_obj.map))

        # check labels
        if Path(atlas_obj.labels_file).is_file():  # if we have a file with indices and labels
            # Logger().info('Using labels from file (' + str(atlas_obj.labels_file) + ')')
            labels_dict = dict()

            with open(atlas_obj.labels_file) as f:
                for line in f:
                    (key, val) = line.split("\t")
                    labels_dict[int(key)] = val

            # check if map indices correspond with indices in the labels file
            if not sorted(atlas_obj.indices) == sorted(list(labels_dict.keys())):
                Logger().info(
                    'The indices in map image ARE NOT the same as those in your *_labels.txt! Ignoring *_labels.txt.')
                Logger().info('MapImage: ')
                Logger().info(sorted(self.indices))
                Logger().info('File: ')
                Logger().info(sorted(list(labels_dict.keys())))

                atlas_obj.roi_list = [RoiObject(index=i, label=str(i), size=np.sum(i == atlas_obj.map)) for i in atlas_obj.indices]
            else:
                for i in range(len(atlas_obj.indices)):
                    roi_index = atlas_obj.indices[i]
                    new_roi = RoiObject(index=roi_index, label=labels_dict[roi_index].replace('\n', ''),
                                        size=np.sum(roi_index == atlas_obj.map))
                    atlas_obj.roi_list.append(new_roi)

        else:  # if we don't have a labels file, we just use str(indices) as labels
            atlas_obj.roi_list = [RoiObject(index=i, label=str(i), size=np.sum(i == atlas_obj.map)) for i in
                                  atlas_obj.indices]

        # check for empty ROIs and extract roi mask
        for roi in atlas_obj.roi_list:

            if roi.size == 0:
                continue
                # Logger().info('ROI with index ' + str(roi.index) + ' and label ' + roi.label + ' does not exist!')

            roi.mask = image.new_img_like(atlas_obj.path, atlas_obj.map == roi.index)
            if target_affine is not None and target_shape is not None:
                roi.mask = image.resample_img(roi.mask, target_affine=target_affine, target_shape=target_shape,
                                              interpolation='nearest')

                # check orientations
                orient_data = ''.join(nib.aff2axcodes(target_affine))
                orient_roi = ''.join(nib.aff2axcodes(roi.mask.affine))
                orient_ok = orient_roi == orient_data
                if not orient_ok:
                    pass
                    # Logger().info('Orientation of mask and data are not the same: '
                    #               + orient_roi + ' (mask) vs. ' + orient_data + ' (data)')

            # check if roi is empty
            if np.sum(roi.mask.dataobj != 0) == 0:
                # Logger().info('No voxels in ROI after resampling (' + roi.label + ').')
                roi.is_empty = True

        if self.loaded_atlasses is None:
            self.loaded_atlasses = dict()
        self.loaded_atlasses[(atlas_name, str(target_affine), str(target_shape))] = atlas_obj
        print("Loading Atlas done!")

    def get_atlas(self, atlas_name, target_affine=None, target_shape=None, mask_threshold=None):

        if self.available_atlasses is None:
            self._inspect_atlas_dir()

        if self.loaded_atlasses is None or (atlas_name, str(target_affine), str(target_shape)) not in self.loaded_atlasses:
            print("Loading Atlas from filesystem")
            self._load_atlas(atlas_name, target_affine, target_shape, mask_threshold)

        return self.loaded_atlasses[(atlas_name, str(target_affine), str(target_shape))]

    @staticmethod
    def find_rois_by_label(atlas_obj, query_list):
        return [i for i in atlas_obj.roi_list if i.label in query_list]

    @staticmethod
    def find_rois_by_index(atlas_obj, query_list):
        return [i for i in atlas_obj.roi_list if i.index in query_list]

    @staticmethod
    def get_nii_files_from_folder(folder_path, extension=".nii.gz"):
        return glob.glob(folder_path + '*' + extension)


class BrainAtlas(BaseEstimator):
    def __init__(self, atlas_name: str, extract_mode: str='vec',
                 mask_threshold=None, background_id=0, rois='all'):

        # ToDo
        # + add-own-atlas capability
        # + get atlas infos (.getInfo('AAL'))
        # + ROIs by name (e.g. in HarvardOxford subcortical: ['Caudate_L', 'Caudate_R'])
        # + get ROIs by map indices (e.g. in AAL: [2001, 2111])
        # + finish mask-img matching
        #   + reorientation/affine transform
        #   + voxel-size (now returns true number of voxels (i.e. after resampling) vs. voxels in mask)
        #   + check RAS vs. LPS view-type and provide warning
        # + handle "disappearing" ROIs when downsampling in map check
        # - pretty getBox function (@Claas)
        # + prettify box-output (to 4d np array)
        # - unit tests
        # Later
        # - add support for overlapping ROIs and probabilistic atlases using 4d-nii
        # - add support for 4d resting-state data using nilearn

        self.atlas_name = atlas_name
        self.extract_mode = extract_mode
        # collection mode default to concat --> can only be overwritten by AtlasMapper
        self.collection_mode = 'concat'
        self.mask_threshold = mask_threshold
        self.background_id = background_id
        self.rois = rois
        self.box_shape = []
        self.is_transformer = True
        self.mask_indices = None
        self.affine = None
        self.shape = None
        self.needs_y = False
        self.needs_covariates = False

    def fit(self, X, y):
        return self

    def transform(self, X, y=None, **kwargs):

        if len(X) < 1:
            raise Exception("Brain Atlas: Did not get any data in parameter X")

        if self.collection_mode == 'list' or self.collection_mode == 'concat':
            collection_mode = self.collection_mode
        else:
            collection_mode = 'concat'
            Logger().error("Collection mode {} not supported. Use 'list' or 'concat' instead."
                           "Falling back to concat mode.".format(self.collection_mode))

        # 1. validate if all X are in the same space and have the same voxelsize and have the same orientation

        # 2. load sample data to get target affine and target shape to adapt the brain atlas

        self.affine, self.shape = BrainMasker.get_format_info_from_first_image(X)

        # load all niftis to memory
        if isinstance(X, list):
            n_subjects = len(X)
            X = image.load_img(X)
        elif isinstance(X, str):
            n_subjects = 1
            X = image.load_img(X)
        elif isinstance(X, np.ndarray):
            n_subjects = X.shape[0]
            X = image.load_img(X)
        else:
            n_subjects = X.shape[-1]

        # get ROI mask
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, self.affine, self.shape, self.mask_threshold)
        roi_objects = self._get_rois(atlas_obj, which_rois=self.rois, background_id=self.background_id)

        roi_data = [list() for i in range(n_subjects)]
        roi_data_concat = list()
        t1 = time.time()

        # convert to series and C ordering since this will speed up the masking process
        series = _utils.as_ndarray(_safe_get_data(X), dtype='float32', order="C", copy=True)
        mask_indices = list()

        for i, roi in enumerate(roi_objects):
            Logger().debug("Extracting ROI {}".format(roi.label))
            # simply call apply_mask to extract one roi
            extraction = self.apply_mask(series, roi.mask)
            if collection_mode == 'list':
                for sub_i in range(extraction.shape[0]):
                    roi_data[sub_i].append(extraction[sub_i])
            else:
                roi_data_concat.append(extraction)
                mask_indices.append(np.ones(extraction[0].size) * i)

        if self.collection_mode == 'concat':
            roi_data = np.concatenate(roi_data_concat, axis=1)
            self.mask_indices = np.concatenate(mask_indices)

        elapsed_time = time.time() - t1
        Logger().debug("Time for extracting {} ROIs in {} subjects: {} seconds".format(len(roi_objects), n_subjects, elapsed_time))
        return roi_data

    def apply_mask(self, series, mask_img):
        mask_img = _utils.check_niimg_3d(mask_img)
        mask, mask_affine = masking._load_mask_img(mask_img)
        mask_img = image.new_img_like(mask_img, mask, mask_affine)
        mask_data = _utils.as_ndarray(mask_img.get_data(),
                                      dtype=np.bool)
        return series[mask_data].T

    def inverse_transform(self, X, y=None, **kwargs):
        # if X.ndim == 2:
        #  Shape: (mask.shape[0], mask.shape[1], mask.shape[2], X.shape[0])
        # todo: we should check that
        X = np.asarray(X)
        # get ROI masks
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, self.affine, self.shape, self.mask_threshold)
        roi_objects = self._get_rois(atlas_obj, which_rois=self.rois, background_id=self.background_id)

        first_mask = roi_objects[0].mask
        unmasked = np.empty((first_mask.shape[0], first_mask.shape[1], first_mask.shape[2]), dtype=X.dtype)
        unmasked[:] = np.nan

        for i, roi in enumerate(roi_objects):
            mask, mask_affine = masking._load_mask_img(roi.mask)
            mask_img = image.new_img_like(roi.mask, mask, mask_affine)
            mask_data = _utils.as_ndarray(mask_img.get_data(), dtype=np.bool)

            unmasked[mask_data] = np.abs(X[self.mask_indices == i])

        return image.new_img_like(first_mask, unmasked)

    @staticmethod
    def _get_rois(atlas_obj, which_rois='all', background_id=0):

        if isinstance(which_rois, str):
            if which_rois == 'all':
                return [roi for roi in atlas_obj.roi_list if roi.index != background_id]
            else:
                return AtlasLibrary().find_rois_by_label(atlas_obj, [which_rois])

        elif isinstance(which_rois, int):
            return AtlasLibrary().find_rois_by_index(atlas_obj, [which_rois])

        elif isinstance(which_rois, list):
            if isinstance(which_rois[0], str):
                if which_rois[0].lower() == 'all':
                    return [roi for roi in atlas_obj.roi_list if roi.index != background_id]
                else:
                    return AtlasLibrary().find_rois_by_label(atlas_obj, which_rois)
            else:
                return AtlasLibrary().find_rois_by_index(atlas_obj, which_rois)


class BrainMasker(BaseEstimator):

    def __init__(self, mask_image=None, affine=None, shape=None, extract_mode='vec'):
        self.mask_image = mask_image
        self.affine = affine
        self.shape = shape
        self.masker = None
        self.extract_mode = extract_mode

    @staticmethod
    def get_format_info_from_first_image(X):
        if isinstance(X, str):
            img = image.load_img(X)
        elif isinstance(X, list) or isinstance(X, np.ndarray):
            if isinstance(X[0], str):
                img = image.load_img(X[0])
            elif isinstance(X[0], nib.Nifti1Image):
                img = X[0]
        elif isinstance(X, nib.Nifti1Image):
            img = X
        else:
            error_msg = "Can only process strings as file paths to nifti images or nifti image object"
            Logger().error(error_msg)
            raise ValueError(error_msg)

        if len(img.shape) > 3:
            img_shape = img.shape[:3]
        else:
            img_shape = img.shape
        return img.affine, img_shape

    @staticmethod
    def _get_box(in_imgs, roi):
        # get ROI infos
        map = roi.get_data()
        true_points = np.argwhere(map)
        corner1 = true_points.min(axis=0)
        corner2 = true_points.max(axis=0)
        box = []
        for img in in_imgs:
            if isinstance(img, str):
                data = image.load_img(img).get_data()
            else:
                data = img.get_data()
            tmp = data[corner1[0]:corner2[0] + 1, corner1[1]:corner2[1] + 1, corner1[2]:corner2[2] + 1]
            box.append(tmp)
        return np.asarray(box)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None, **kwargs):

        if self.affine is None or self.shape is None:
            self.affine, self.shape = BrainMasker.get_format_info_from_first_image(X)

        if isinstance(self.mask_image, str):
            self.mask_image = MaskLibrary().get_mask(self.mask_image, self.affine, self.shape)
        elif isinstance(self.mask_image, RoiObject):
            pass

        if not self.mask_image.is_empty:
            self.masker = NiftiMasker(mask_img=self.mask_image.mask, target_affine=self.affine,
                                      target_shape=self.shape, dtype='float32')
            try:
                single_roi = self.masker.fit_transform(X)
            except BaseException as e:
                Logger().error(e)
                single_roi = None

            if single_roi is not None:
                if self.extract_mode == 'vec':
                    return np.asarray(single_roi)

                elif self.extract_mode == 'mean':
                    return np.mean(single_roi, axis=1)

                elif self.extract_mode == 'box':
                    return BrainMasker._get_box(X, self.mask_image)

                elif self.extract_mode == 'img':
                    return self.masker.inverse_transform(single_roi)

                else:
                    Logger().error("Currently there are no other methods than 'vec', 'mean', and 'box' supported!")
            else:
                print("Extracting ROI failed.")
        else:
            print("Skipping self.mask_image " + self.mask_image.label + " because it is empty.")


