import numpy as np
from nilearn.image import load_img
from nilearn.image import new_img_like
from pathlib import Path
from sklearn.base import BaseEstimator

class BrainAtlas(BaseEstimator):
    def __init__(self, atlas_name=None, extract_mode='mean', whichROIs='all', background_id=0):
        # ToDo
        # + add-own-atlas capability
        # + get boxes from ROIs
        # + get atlas infos (.getInfo('AAL'))
        # + get list of available atlases (BrainAtlas('?'))
        # + get all ROIs ('all'),
        # + ROIs by name (e.g. in HarvardOxford subcortical: ['Caudate_L', 'Caudate_R'])
        # + get ROIs by map indices (e.g. in AAL: [2001, 2111])
        # - finish mask-img matching
        #   + reorientation/affine transform
        #   + voxel-size
        #   - RAS vs. LPS view-type
        # - handle "disappearing" ROIs when downsampling in map check
        # - unit tests
        # Later
        # - add support for overlapping ROIs and probabilistic atlases using 4d-nii
        # - add support for 4d resting-state data using nilearn

        self.extract_mode = extract_mode
        self.whichROIs = whichROIs
        self.background_id = background_id

        # get info about available atlases
        ATLAS_DICT, atlas_dir = BrainAtlas._getAtlasDict()

        # get Atlas
        self.atlas_name = atlas_name
        self.atlas_path = ATLAS_DICT[self.atlas_name][0]
        self.map = load_img(self.atlas_path).get_data()  # get actual map data
        self.indices = list(np.unique(self.map))

        # check labels
        if Path(ATLAS_DICT[self.atlas_name][1]).is_file(): # if we have a file with indices and labels
            print('You have a labels file ( ' + ATLAS_DICT[self.atlas_name][1] + '). Yeah, using labels...')
            labels_dict = dict()

            with open(ATLAS_DICT[self.atlas_name][1]) as f:
                for line in f:
                    (key, val) = line.split("\t")
                    labels_dict[int(key)] = val

            # check if map indices correspond with indices in the labels file
            if not sorted(self.indices) == sorted(list(labels_dict.keys())):
                print('The indices in your map image ARE NOT the same as those in your *_labels.txt! Ignoring *_labels.txt.')
                print('MapImage: ' + sorted(self.indices))
                print('File: ' + sorted(list(labels_dict.keys())))
                self.labels = list(str(i) for i in self.indices)
            else:
                self.labels = []
                for i in range(len(self.indices)):
                    self.labels.append(labels_dict[self.indices[i]].replace('\n', ''))

        else: # if we don't have a labels file, we just use str(indices) as labels
            self.labels = list(str(i) for i in self.indices)

        self.roi_sizes = [np.sum(i == self.map) for i in np.unique(self.indices)] # number of voxels per ROI
        # check for empty ROIs
        for i in range(len(self.roi_sizes)):
            if self.roi_sizes[i] == 0:
                print('ROI with index ' + str(self.indices[i]) + ' and label ' + self.labels[i] + ' does not exist!')

        self.box_shape = []
        self.gotData = False

    def fit(self):
        pass

    def transform(self, X, y=None):
        extract_mode = self.extract_mode
        whichROIs = self.whichROIs
        background_id = self.background_id
        self.gotData = True

        # get ROI infos
        rois = self._getROIs(whichROIs=whichROIs, background_id=background_id)

        # Grab masker and apply to structural data for each ROI
        from nilearn.input_data import NiftiMasker
        from nilearn import image
        img = load_img(X[0])
        roi_data = []
        if extract_mode == 'box':
            self.box_shape = []
        for roi in rois:
            roi = image.resample_img(roi, target_affine=img.affine, target_shape=img.shape, interpolation='nearest')

            masker = NiftiMasker(mask_img=roi, target_affine=img.affine, target_shape=img.shape)
            single_roi = masker.fit_transform(X)
            if extract_mode == 'vec':
                roi_data.append(single_roi)
            elif extract_mode == 'mean':
                tmp = []
                for sample_ind in range(len(single_roi)):
                    tmp.append(np.mean(single_roi[sample_ind]))
                roi_data.append(tmp)
            elif extract_mode == 'box':
                tmp = []
                tmp, bshape = self._getBox(X, roi)
                roi_data.append(tmp)
                self.box_shape.append(bshape)
            else:
                # any function which can work on a vector passed as a string
                tmp = []
                # find something safer than eval!
                any_func = lambda ex, data, opt_args=None: eval(ex)(data, opt_args)
                expr = extract_mode
                for sample_ind in range(0, len(single_roi)):
                    t = any_func(expr, single_roi[sample_ind])
                    tmp.append(t)
                roi_data.append(tmp)

        return roi_data

    def _getROIs(self, whichROIs='all', background_id=0):
        if whichROIs == 'all': # just use all
            # remove background
            self.indices_applied = []
            for s in self.indices:
                if s != 0:
                    self.indices_applied.append(s)
            # get labels_applied and roi_sizes_applied for indices (excluding background)
            self.labels_applied = [self.labels[self.indices.index(i)] for i in self.indices_applied]
            self.roi_sizes_applied = [self.roi_sizes[self.labels.index(i)] for i in self.labels_applied]

        elif isinstance(whichROIs, list):
            if all(isinstance(item, int) for item in whichROIs): # use list of indices in map (ints)
                #self.indices_applied = whichROIs
                self.labels_applied = [self.labels[self.indices.index(i)] for i in whichROIs]
                self.indices_applied = [self.indices[self.labels.index(i)] for i in self.labels_applied]
                self.roi_sizes_applied = [self.roi_sizes[self.labels.index(i)] for i in self.labels_applied]

            elif all(isinstance(item, str) for item in whichROIs): # use list of labels (strings)
                #self.labels_applied = whichROIs
                self.indices_applied = [self.indices[self.labels.index(i)] for i in whichROIs]
                self.labels_applied = [self.labels[self.indices.index(i)] for i in self.indices_applied]
                self.roi_sizes_applied = [self.roi_sizes[self.labels.index(i)] for i in self.labels_applied]
        else:
            print('Pass a list of indices (ints) or ROI names (strings). Or use all via "all".')

        # collect roi masks
        rois = []
        for ind in self.indices_applied:
            if ind == background_id:
                continue
            rois.append(new_img_like(self.atlas_path, self.map == ind))

        return rois

    def _getBox(self, in_imgs, roi):
        # get ROI infos
        map = roi.get_data()
        true_points = np.argwhere(map)
        corner1 = true_points.min(axis=0)
        corner2 = true_points.max(axis=0)
        box = []
        for img in in_imgs:
            data = load_img(img).get_data()
            tmp = data[corner1[0]:corner2[0] + 1, corner1[1]:corner2[1] + 1, corner1[2]:corner2[2] + 1]
            box.append(tmp)
        #box = np.asarray(box)
        return box, tmp.shape

    def _adjustAtlas(self):
        # adjust atlas to input imgs regarding reorientation (affine transform), voxel-size, view type (e.g. LPS vs. RAS)
        pass

    @staticmethod
    def _getAtlasDict():
        # Get which atlases are available in Atlases subdir of this module
        # all have to be in *.(nii/img).gz! (Could be nicer)
        import os.path
        import inspect
        import glob

        atlas_dir = os.path.dirname(inspect.getfile(BrainAtlas)) + '/' + 'Atlases/'
        atlas_files = glob.glob(atlas_dir + '*/*.nii.gz')
        ATLAS_DICT = dict()
        for atlas_file in atlas_files:
            atlas_id = os.path.basename(atlas_file)[:-7]
            ATLAS_DICT[atlas_id] = [atlas_file, (atlas_file[:-7] + '_labels.txt')]

        return ATLAS_DICT, atlas_dir

    @staticmethod
    def whichAtlases():
        # get info about available atlases
        print('\nAvailable Atlases:')
        ATLAS_DICT, atlas_dir = BrainAtlas._getAtlasDict()
        for key in ATLAS_DICT.keys():
            print("'" + key + "'")
        print('\nCopy your favorite atlas as a *.gz file (e.g. myFavAtlas.nii.gz) to ' + atlas_dir + ' and enjoy.')
        print('Add a text file containing ROI values and labels (e.g. myFavAtlas_labels.txt) to be able to use your ROI labels with Photon.')
        return ATLAS_DICT.keys()

    def getInfo(self):
        print('\nAtlas Name: ' + self.atlas_name)
        if self.gotData == False:
            print('#ROIs: ' + str(len(np.unique(self.indices))))
            print('#\tROI Index\tROI Label\tROI Size')
            for i in range(len(self.indices)):
                print(str(i + 1) + '\t' + str(self.indices[i]) + '\t' + self.labels[i] + '\t' + str(
                    self.roi_sizes[i]))
        else:
            print('#ROIs applied: ' + str(len(self.indices_applied)))
            if self.box_shape == []:
                print('#\tROI Index\tROI Label\tROI Size')
                for i in range(len(self.indices_applied)):
                    print(str(i + 1) + '\t' + str(self.indices_applied[i]) + '\t' + self.labels_applied[i] + '\t' + str(
                        self.roi_sizes_applied[i]))
            else:
                print('#\tROI Index\tROI Label\tROI Size\tBox Shape\tBox Size\t% ROI Voxels in Box')
                from operator import mul
                import functools
                for i in range(len(self.indices_applied)):
                    box_prod = functools.reduce(mul, self.box_shape[i])
                    print(str(i + 1) + '\t' + str(self.indices_applied[i]) + '\t' + self.labels_applied[i] + '\t' + str(
                        self.roi_sizes_applied[i]) + '\t' + str(self.box_shape[i]) + '\t' + str(box_prod) + '\t' +
                          str("%.0f" % (self.roi_sizes_applied[i] / box_prod * 100)) + '%')


# if __name__ == '__main__':
#
#
#     from nilearn import datasets
#     dataset_files = datasets.fetch_oasis_vbm(n_subjects=5)
#     from nilearn.datasets import MNI152_FILE_PATH
#     dataset_files = [MNI152_FILE_PATH, MNI152_FILE_PATH]
#
#     availbleAtlases = BrainAtlas.whichAtlases() # get list of available atlases and help.
#     extMeth = ['mean', 'vec', 'box', 'np.std']
#     roi_data = []
#     for atlas in availbleAtlases:
#         for em in extMeth:
#             print(atlas + ': ' + em)
#             myAtlas = BrainAtlas(atlas_name=atlas, extract_mode=em, whichROIs='all')
#             #myAtlas = BrainAtlas(atlas_name='AAL', extract_mode='vec', whichROIs=[2001, 2111, 6301])
#             #myAtlas = BrainAtlas(atlas_name='AAL', extract_mode='box', whichROIs=['Frontal_Sup_R', 'Caudate_L', 'Temporal_Inf_R'])
#
#             #myAtlas.getInfo()
#             roi_data.append(myAtlas.transform(X=dataset_files)) # ROI indices
#
#             #myAtlas.getInfo()
#     print('')
#

    # from nilearn.datasets import MNI152_FILE_PATH
    # nii_file = MNI152_FILE_PATH
    #
    # #from nilearn import plotting
    # #plotting.plot_glass_brain(nii_file)
    # #plotting.show()
