import unittest, os, inspect
from ..neuro.BrainAtlas import AtlasLibrary, BrainAtlas, BrainMasker
from ..neuro.NeuroBase import NeuroModuleBranch
from ..base.PhotonPipeline import CacheManager
from ..base.PhotonBase import PipelineElement
from ..neuro.ImageBasics import ResampleImages, SmoothImages
from nilearn import image
from nilearn.input_data import NiftiMasker
from nibabel.nifti1 import Nifti1Image
import numpy as np
import glob
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.validation.ResultsTreeHandler import ResultsHandler
from sklearn.model_selection import ShuffleSplit


class NeuroTest(unittest.TestCase):

    def setUp(self):
        self.test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/')
        self.atlas_name = "AAL"
        self.roi_list = ["Hippocampus_R", "Hippocampus_L", "Amygdala_L", "Amygdala_R"]
        self.X = AtlasLibrary().get_nii_files_from_folder(self.test_folder, extension=".nii")
        self.y = np.random.randn(len(self.X))

    def tearDown(self):
        pass

    def test_single_subject_resampling(self):
        voxel_size = [3, 3, 3]

        # nilearn
        from nilearn.image import resample_img
        nilearn_resampled_img = resample_img(self.X[0], interpolation='nearest', target_affine = np.diag(voxel_size))
        nilearn_resampled_array = nilearn_resampled_img.dataobj

        # photon
        resampler = PipelineElement('ResampleImages', hyperparameters={}, voxel_size=voxel_size, batch_size=1)
        single_resampled_img, _, _ = resampler.transform(self.X[0])

        branch = NeuroModuleBranch('NeuroBranch')
        branch += resampler
        branch_resampled_img, _, _ = branch.transform(self.X[0])

        # assert
        self.assertIsInstance(single_resampled_img, np.ndarray)
        self.assertIsInstance(branch_resampled_img, Nifti1Image)

        self.assertTrue(np.array_equal(nilearn_resampled_array, single_resampled_img))
        self.assertTrue(np.array_equal(single_resampled_img, branch_resampled_img.dataobj))

    def test_multi_subject_resampling(self):
        voxel_size = [3, 3, 3]

        # nilearn
        from nilearn.image import resample_img, index_img
        nilearn_resampled = resample_img(self.X[:3], interpolation='nearest', target_affine = np.diag(voxel_size))
        nilearn_resampled_img = [index_img(nilearn_resampled, i) for i in range(nilearn_resampled.shape[-1])]
        nilearn_resampled_array = np.moveaxis(nilearn_resampled.dataobj, -1, 0)

        # photon
        resampler = PipelineElement('ResampleImages', hyperparameters={}, voxel_size=voxel_size)
        resampled_img, _, _ = resampler.transform(self.X[:3])

        branch = NeuroModuleBranch('NeuroBranch')
        branch += resampler
        branch_resampled_img, _, _ = branch.transform(self.X[:3])

        # assert
        self.assertIsInstance(resampled_img, np.ndarray)
        self.assertIsInstance(branch_resampled_img, list)
        self.assertIsInstance(branch_resampled_img[0], Nifti1Image)

        self.assertTrue(np.array_equal(nilearn_resampled_array, resampled_img))
        self.assertTrue(np.array_equal(branch_resampled_img[1].dataobj, nilearn_resampled_img[1].dataobj))

    def test_single_subject_smoothing(self):

        # nilearn
        from nilearn.image import smooth_img
        nilearn_smoothed_img = smooth_img(self.X[0], fwhm=[3, 3, 3])
        nilearn_smoothed_array = nilearn_smoothed_img.dataobj

        # photon
        smoother = PipelineElement('SmoothImages', hyperparameters={}, fwhm=3, batch_size=1)
        photon_smoothed_array, _, _ = smoother.transform(self.X[0])

        branch = NeuroModuleBranch('NeuroBranch')
        branch += smoother
        photon_smoothed_img, _, _ = branch.transform(self.X[0])

        # assert
        self.assertIsInstance(photon_smoothed_array, np.ndarray)
        self.assertIsInstance(photon_smoothed_img, Nifti1Image)

        self.assertTrue(np.array_equal(photon_smoothed_array, nilearn_smoothed_array))
        self.assertTrue(np.array_equal(photon_smoothed_img.dataobj, nilearn_smoothed_img.dataobj))

    def test_multi_subject_smoothing(self):
        # nilearn
        from nilearn.image import smooth_img
        nilearn_smoothed_img = smooth_img(self.X[0:3], fwhm=[3, 3, 3])
        nilearn_smoothed_array = nilearn_smoothed_img[1].dataobj

        # photon
        smoother = PipelineElement('SmoothImages', hyperparameters={}, fwhm=3)
        photon_smoothed_array, _, _ = smoother.transform(self.X[0:3])


        branch = NeuroModuleBranch('NeuroBranch')
        branch += smoother
        photon_smoothed_img, _, _ = branch.transform(self.X[0:3])

        # assert
        self.assertIsInstance(photon_smoothed_array, np.ndarray)
        self.assertIsInstance(photon_smoothed_img[0], Nifti1Image)

        self.assertTrue(np.array_equal(photon_smoothed_array[1], nilearn_smoothed_array))
        self.assertTrue(np.array_equal(photon_smoothed_img[1].dataobj, nilearn_smoothed_img[1].dataobj))

    def test_brain_atlas_load(self):

        brain_atlas = AtlasLibrary().get_atlas(self.atlas_name)

        # manually load brain atlas
        man_map = image.load_img(os.path.dirname(inspect.getfile(BrainAtlas)) + '/Atlases/AAL_SPM12/AAL.nii.gz').get_data()
        self.assertTrue(np.array_equal(man_map, brain_atlas.map))

    def test_brain_masker(self):

        affine, shape = BrainMasker.get_format_info_from_first_image(self.X)
        atlas_obj = AtlasLibrary().get_atlas(self.atlas_name, affine, shape)
        roi_objects = BrainAtlas._get_rois(atlas_obj, which_rois=self.roi_list, background_id=0)

        for roi in roi_objects:
            masker = BrainMasker(mask_image=roi, affine=affine, shape=shape, extract_mode="vec")
            own_calculation = masker.transform(self.X[0])
            nilearn_func = NiftiMasker(mask_img=roi.mask, target_affine=affine, target_shape=shape, dtype='float32')
            nilearn_calculation = nilearn_func.fit_transform(self.X[0])

            self.assertTrue(np.array_equal(own_calculation, nilearn_calculation))

    def test_brain_atlas(self):

        brain_atlas = BrainAtlas(self.atlas_name, "vec", rois=self.roi_list)
        new_data = brain_atlas.transform(self.X)
        self.assertTrue(len(self.X), len(brain_atlas.rois))

        brain_atlas_mean = BrainAtlas(self.atlas_name, "mean", rois='all')
        brain_atlas_mean.transform(self.X)
        # Todo: how to compare?
        debug = True

    def test_resampling_and_smoothing(self):

        testsuite = ["Testing Method on Single Core",
                     "Testing Method on Multi Core",
                     "Testing Method on Single Core Batched",
                     "Testing Method on Multi Core Batched"]

        def create_instances_and_transform(neuro_class_str, param_dict, transformed_X):
            instance_list = []

            nmb1 = NeuroModuleBranch(name="single core application", nr_of_processes=1)

            nmb1.add(PipelineElement(neuro_class_str, **param_dict))
            instance_list.append(nmb1)

            nmb2 = NeuroModuleBranch(name="multi core application", nr_of_processes=3)
            nmb2.add(PipelineElement(neuro_class_str, **param_dict))
            instance_list.append(nmb2)

            nmb3 = NeuroModuleBranch(name="batched single core application", nr_of_processes=1)
            nmb3.add(PipelineElement(neuro_class_str, batch_size=5, **param_dict))
            instance_list.append(nmb3)

            nmb4 = NeuroModuleBranch(name="batched multi core application", nr_of_processes=3)
            nmb4.add(PipelineElement(neuro_class_str, batch_size=5, **param_dict))
            instance_list.append(nmb4)

            for test, obj in enumerate(instance_list):
                print(testsuite[test])

                # transform data
                obj.base_element.cache_folder = os.path.join(self.test_folder, 'cache')
                obj.base_element.current_config = {'test_suite': 1}
                new_X, _, _ = obj.transform(self.X)
                obj.base_element.clear_cache()

                # compare output to nilearn version
                for index, nilearn_nifti in enumerate(transformed_X):
                    photon_nifti = new_X[index]
                    if isinstance(photon_nifti, Nifti1Image):
                        self.assertTrue(np.array_equal(photon_nifti.dataobj, nilearn_nifti.dataobj))
                    else:
                        self.assertTrue(np.array_equal(np.asarray(photon_nifti), nilearn_nifti.dataobj))

                print("finished testing object: all images are fine.")

        print("Testing Nifti Smoothing.")
        smoothing_param_dict = {'fwhm': [3, 3, 3]}
        nilearn_smoothed_X = []
        for element in self.X:
            nilearn_smoothed_X.append(image.smooth_img(element, **smoothing_param_dict))
        create_instances_and_transform('SmoothImages', smoothing_param_dict, nilearn_smoothed_X)

        print("Testing Nifti Resampling.")
        target_affine = np.diag([5, 5, 5])
        resample_param_dict = {'target_affine': target_affine, 'interpolation': 'nearest'}
        nilearn_resampled_X = []
        for element in self.X:
            nilearn_resampled_X.append(image.resample_img(element, **resample_param_dict))
        create_instances_and_transform('ResampleImages', {'voxel_size': [5, 5, 5]}, nilearn_resampled_X)

    def test_neuro_module_branch(self):
        nmb = NeuroModuleBranch('best_branch_ever')
        nmb += PipelineElement('SmoothImages', fwhm=10)
        nmb += PipelineElement('ResampleImages', voxel_size=5)
        nmb += PipelineElement('BrainAtlas', rois=['Hippocampus_L', 'Hippocampus_R'],
                               atlas_name="AAL", extract_mode='vec')

        nmb.base_element.cache_folder = os.path.join(self.test_folder, 'cache')
        CacheManager.clear_cache_files(nmb.base_element.cache_folder, True)
        # set the config so that caching works
        nmb.set_params(**{'SmoothImages__fwhm': 10, 'ResampleImages__voxel_size': 5})

        # okay we are transforming 8 Niftis with 3 elements, so afterwards there should be 3*8 + 1 for library
        nr_niftis = 7
        nmb.transform(self.X[:nr_niftis])
        nr_files_in_folder = len(glob.glob(os.path.join(nmb.base_element.cache_folder, "*.p")))
        self.assertTrue(nr_files_in_folder == (3*nr_niftis) + 1)
        self.assertTrue(len(nmb.base_element.cache_man.cache_index.items()) == (3*nr_niftis))

        # transform 3 items that should have been cached and two more that need new processing
        nmb.transform(self.X[nr_niftis-2::])
        # now we should have 10 * 3 + 1 elements in the cache folder
        nr_files_in_folder = len(glob.glob(os.path.join(nmb.base_element.cache_folder, "*.p")))
        self.assertTrue(nr_files_in_folder == (3 * len(self.X)) + 1)
        self.assertTrue(len(nmb.base_element.cache_man.cache_index.items()) == (3 * len(self.X)))

    def test_custom_mask(self):
        custom_mask = 'photonai/neuro/Atlases/Cerebellum/P_08_Cere.nii.gz'
        mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec', batch_size=20)
        X_masked = mask.transform(self.X)

        with self.assertRaises(FileNotFoundError):
            mask = PipelineElement('BrainMask', mask_image='XXXXX', extract_mode='vec', batch_size=20)
            mask.transform(self.X)

    def test_custom_atlas(self):
        custom_atlas = 'photonai/neuro/Atlases/AAL_SPM12/AAL.nii.gz'

        atlas = PipelineElement('BrainAtlas', atlas_name=custom_atlas, extract_mode='vec', batch_size=20)
        X_masked = atlas.transform(self.X)

        with self.assertRaises(FileNotFoundError):
            atlas = PipelineElement('BrainAtlas', atlas_name='XXXXX', extract_mode='vec', batch_size=20)
            atlas.transform(self.X)

    def test_atlas_mapper(self):
        pass

    def test_inverse_transform(self):
        settings = OutputSettings(project_folder='test_results',
                                  save_feature_importances='best',
                                  overwrite_results=True)

        # DESIGN YOUR PIPELINE
        pipe = Hyperpipe('Limbic_System',
                         optimizer='grid_search',
                         metrics=['mean_absolute_error'],
                         best_config_metric='mean_absolute_error',
                         outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                         inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                         verbosity=2,
                         cache_folder="./cache",
                         eval_final_performance=True,
                         output_settings=settings)

        # PICK AN ATLAS
        atlas = PipelineElement('BrainAtlas',
                                rois=['Hippocampus_L', 'Amygdala_L'],
                                atlas_name="AAL", extract_mode='vec', batch_size=20)

        # EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
        neuro_branch = NeuroModuleBranch('NeuroBranch')
        neuro_branch += atlas
        pipe += neuro_branch

        pipe += PipelineElement('LinearSVR')

        pipe.fit(self.X, self.y)

        # GET IMPORTANCE SCORES
        handler = ResultsHandler(pipe.results)
        importance_scores_optimum_pipe = handler.results.optimum_pipe_feature_importances

        manual_img, _, _ = pipe.optimum_pipe.inverse_transform(importance_scores_optimum_pipe, None)
        img = image.load_img('test_results/Limbic_System_results/optimum_pipe_feature_importances_backmapped.nii.gz')
        self.assertTrue(np.array_equal(manual_img.get_data(), img.get_data()))

    def test_all_atlases(self):
        for atlas in AtlasLibrary().ATLAS_DICTIONARY.keys():
            brain_atlas = PipelineElement('BrainAtlas', atlas_name=atlas, extract_mode='vec')
            brain_atlas.transform(self.X)

    def test_all_masks(self):
        for mask in AtlasLibrary().MASK_DICTIONARY.keys():
            brain_mask = PipelineElement('BrainMask', mask_image=mask, extract_mode='vec')
            brain_mask.transform(self.X)



