import numpy as np
from nilearn.image import load_img
from nilearn.image import new_img_like
from pathlib import Path
from sklearn.base import BaseEstimator
from ..Logging.Logger import Logger


class BrainAtlas(BaseEstimator):
    def __init__(self, atlas_info_object):

        # ToDo
        # + add-own-atlas capability
        # + get atlas infos (.getInfo('AAL'))
        # + get list of available atlases (BrainAtlas.whichAtlases())
        # + get all ROIs ('all'),
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

        self.atlas_info_object = atlas_info_object
        self.extract_mode = atlas_info_object.extraction_mode
        self.whichROIs = atlas_info_object.roi_names
        self.background_id = atlas_info_object.background_id
        self.mask_threshold = atlas_info_object.mask_threshold

        # get info about available atlases
        ATLAS_DICT, atlas_dir = BrainAtlas._getAtlasDict()

        # get Atlas
        self.atlas_name = atlas_info_object.atlas_name
        self.atlas_path = ATLAS_DICT[self.atlas_name][0]
        if self.mask_threshold is None:
            self.map = load_img(self.atlas_path).get_data()  # get actual map data
        else:
            self.map = load_img(self.atlas_path).get_data() > self.mask_threshold  # get actual map data with probability maps
            self.map = self.map.astype(int)

        self.indices = list(np.unique(self.map))

        # check labels
        if Path(ATLAS_DICT[self.atlas_name][1]).is_file(): # if we have a file with indices and labels
            Logger().info('Using labels from file (' + ATLAS_DICT[self.atlas_name][1] + ')')
            labels_dict = dict()

            with open(ATLAS_DICT[self.atlas_name][1]) as f:
                for line in f:
                    (key, val) = line.split("\t")
                    labels_dict[int(key)] = val

            # check if map indices correspond with indices in the labels file
            if not sorted(self.indices) == sorted(list(labels_dict.keys())):
                Logger().info('The indices in map image ARE NOT the same as those in your *_labels.txt! Ignoring *_labels.txt.')
                Logger().info('MapImage: ')
                Logger().info(sorted(self.indices))
                Logger().info('File: ')
                Logger().info(sorted(list(labels_dict.keys())))
                #self.labels = list(str(i) for i in self.indices)
                self.labels = []
                for i in range(len(self.indices)):
                    self.labels.append(labels_dict[self.indices[i]].replace('\n', ''))
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
                Logger().info('ROI with index ' + str(self.indices[i]) + ' and label ' + self.labels[i] + ' does not exist!')

        self.box_shape = []
        self.gotData = False

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):

        # if self.last_hash is not None:
        #     current_hash = hash(X)
        #     if self.last_hash == current_hash:
        #         return self.last_transformed_X
        #     else:
        #         # do normal stuff
        #         self.last_hash = current_hash
        #         return usual_variable


        extract_mode = self.extract_mode
        whichROIs = self.whichROIs
        background_id = self.background_id
        self.gotData = True

        # get ROI infos
        rois = self._getROIs(whichROIs=whichROIs, background_id=background_id)

        # Grab masker and apply to structural data for each ROI
        from nilearn.input_data import NiftiMasker
        from nilearn import image

        try:
            img = load_img(X[0])
        except:
            img = X[0]

        import nibabel as nib
        orient_data = ''.join(nib.aff2axcodes(img.affine))

        roi_data = []
        if extract_mode == 'box':
            self.box_shape = []
        i = 0
        out_ind = ()
        for roi in rois:
            roi = image.resample_img(roi, target_affine=img.affine, target_shape=img.shape, interpolation='nearest')

            # check orientations
            orient_roi = ''.join(nib.aff2axcodes(roi.affine))
            orient_ok = orient_roi==orient_data
            if not orient_ok:
                Logger().info('Orientation of mask and data are not the same: ' + orient_roi + ' (mask) vs. ' + orient_data + ' (data)')
                break

            # handle empty ROIs
            if np.sum(roi.dataobj != 0) == 0:
                Logger().info('No voxels in ROI (' + self.labels_applied[i] + ').')
                out_ind = np.append(out_ind, i)
                i += 1
                continue
            else:
                masker = NiftiMasker(mask_img=roi, target_affine=img.affine, target_shape=img.shape)
                try:
                    single_roi = masker.fit_transform(X)
                except BaseException as e:
                    Logger().info(e)

                self.roi_sizes_applied[i] = single_roi.shape[1]
                Logger().info('Extracting data from ' + self.labels_applied[i] + ' (Index: '
                      + str(self.indices_applied[i]) + '; ROI Size: ' + str(self.roi_sizes_applied[i]) + ')')
                i += 1

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
                    # ToDo
                    # find something safer than eval!
                    any_func = lambda ex, data, opt_args=None: eval(ex)(data, opt_args)
                    expr = extract_mode
                    for sample_ind in range(0, len(single_roi)):
                        t = any_func(expr, single_roi[sample_ind])
                        tmp.append(t)
                    roi_data.append(tmp)

        # delete empty ROI info
        self.roi_sizes_applied = [k for j, k in enumerate(self.roi_sizes_applied) if j not in out_ind]
        self.indices_applied = [k for j, k in enumerate(self.indices_applied) if j not in out_ind]
        self.labels_applied = [k for j, k in enumerate(self.labels_applied) if j not in out_ind]

        self.atlas_info_object.roi_names_runtime = self.labels_applied

        if len(roi_data)==1:
            roi_data = roi_data[0]

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
            #Todo: Catch empty list
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
            Logger().info('Pass a list of indices (ints) or ROI names (strings). Or use all via "all".')

        # collect roi masks
        rois = []
        for ind in self.indices_applied:
            if ind == background_id:
                continue
            rois.append(new_img_like(self.atlas_path, self.map == ind))

        self.atlas_info_object.roi_names_runtime = self.labels_applied
        return rois

    @staticmethod
    def _getBox(in_imgs, roi):
        # get ROI infos
        map = roi.get_data()
        true_points = np.argwhere(map)
        corner1 = true_points.min(axis=0)
        corner2 = true_points.max(axis=0)
        box = []
        for img in in_imgs:
            try:
                data = load_img(img).get_data()
            except:
                data = img.get_data()


            tmp = data[corner1[0]:corner2[0] + 1, corner1[1]:corner2[1] + 1, corner1[2]:corner2[2] + 1]
            box.append(tmp)
        #box = np.asarray(box)
        return np.asarray(box), tmp.shape

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
        Logger().info('\nAvailable Atlases:')
        ATLAS_DICT, atlas_dir = BrainAtlas._getAtlasDict()
        for key in ATLAS_DICT.keys():
            Logger().info("'" + key + "'")
        Logger().info('\nCopy your favorite atlas as a *.gz file (e.g. myFavAtlas.nii.gz) to ' + atlas_dir + ' and enjoy.')
        Logger().info('Add a text file containing ROI values and labels (e.g. myFavAtlas_labels.txt) to be able to use your ROI labels with Photon.')
        return ATLAS_DICT.keys()

    def getInfo(self):
        Logger().info('\nAtlas Name: ' + self.atlas_name)
        if not self.gotData:
            Logger().info('#ROIs: ' + str(len(np.unique(self.indices))))
            Logger().info('#\tROI Index\tROI Label\tROI Size')
            for i in range(len(self.indices)):
                Logger().info(str(i + 1) + '\t' + str(self.indices[i]) + '\t' + self.labels[i] + '\t' + str(
                    self.roi_sizes[i]))
        else:
            Logger().info('#ROIs applied: ' + str(len(self.indices_applied)))
            if not self.box_shape:
                Logger().info('#\tROI Index\tROI Label\tROI Size')
                for i in range(len(self.indices_applied)):
                    Logger().info(str(i + 1) + '\t' + str(self.indices_applied[i]) + '\t' + self.labels_applied[i] + '\t' + str(
                        self.roi_sizes_applied[i]))
            else:
                Logger().info('#\tROI Index\tROI Label\tROI Size\tBox Shape\tBox Size\t% ROI Voxels in Box')
                from operator import mul
                import functools
                for i in range(len(self.indices_applied)):
                    box_prod = functools.reduce(mul, self.box_shape[i])
                    Logger().info(str(i + 1) + '\t' + str(self.indices_applied[i]) + '\t' + self.labels_applied[i] + '\t' + str(
                        self.roi_sizes_applied[i]) + '\t' + str(self.box_shape[i]) + '\t' + str(box_prod) + '\t' +
                          str("%.0f" % (self.roi_sizes_applied[i] / box_prod * 100)) + '%')


# if __name__ == '__main__':
#
#
#     # from nilearn import datasets
#     # dataset_files = datasets.fetch_oasis_vbm(n_subjects=5)
#     from nilearn.datasets import MNI152_FILE_PATH
#     dataset_files = [MNI152_FILE_PATH, MNI152_FILE_PATH]
#     Logger().info('')
#
#     availableAtlases = BrainAtlas.whichAtlases()
#
#
#     # get list of available atlases and help.
#     extMeth = ['box', 'vec', 'mean', 'np.std']
#     roi_data = []
#     for em in extMeth:
#         for atlas in availableAtlases:
#             Logger().info('\n\n' + atlas + ': ' + em)
#             myAtlas = BrainAtlas(atlas_name=atlas, extract_mode=em, whichROIs='all')
#             myAtlas.getInfo()
#             roi_data.append(myAtlas.transform(X=dataset_files)) # ROI indices
#             myAtlas.getInfo()
#
#         Logger().info('\n\n' + em)
#         myAtlas = BrainAtlas(atlas_name='AAL', extract_mode=em, whichROIs=[2001, 2111, 6301])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files))  # ROI indices
#         myAtlas.getInfo()
#
#         myAtlas = BrainAtlas(atlas_name='AAL', extract_mode=em, whichROIs=['Frontal_Sup_R', 'Caudate_L', 'Temporal_Inf_R'])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files)) # ROI indices
#         myAtlas.getInfo()
#
#         myAtlas = BrainAtlas(atlas_name='HarvardOxford-cort-maxprob-thr50', extract_mode=em, whichROIs=[1, 29, 8])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files))  # ROI indices
#         myAtlas.getInfo()
#
#         myAtlas = BrainAtlas(atlas_name='HarvardOxford-cort-maxprob-thr50', extract_mode=em, whichROIs=['Superior Temporal Gyrus, anterior division', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', "Heschl's Gyrus (includes H1 and H2)"])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files)) # ROI indices
#         myAtlas.getInfo()
#
#         myAtlas = BrainAtlas(atlas_name='HarvardOxford-sub-maxprob-thr50', extract_mode=em, whichROIs=[10, 11, 12])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files))  # ROI indices
#         myAtlas.getInfo()
#
#         myAtlas = BrainAtlas(atlas_name='HarvardOxford-sub-maxprob-thr50', extract_mode=em, whichROIs=['Left Accumbens', 'Right Accumbens', 'Right Caudate'])
#         #myAtlas.getInfo()
#         roi_data.append(myAtlas.transform(X=dataset_files)) # ROI indices
#         myAtlas.getInfo()
#
#     myAtlas = BrainAtlas(atlas_name='AAL', extract_mode='box', whichROIs='all')
#     myAtlas.getInfo()
#     roi_data = myAtlas.transform(X=dataset_files)  # ROI indices
#     myAtlas.getInfo()
#
#
# Logger().info('')


    # from nilearn.datasets import MNI152_FILE_PATH
    # nii_file = MNI152_FILE_PATH
    #
    # #from nilearn import plotting
    # #plotting.plot_glass_brain(nii_file)
    # #plotting.show()
