import unittest, os, inspect
from ..neuro.BrainAtlas import AtlasLibrary, BrainAtlas
from ..neuro.NeuroBase import NeuroModuleBranch
from ..base.PhotonBatchElement import PhotonBatchElement
from ..base.PhotonBase import PipelineElement
from ..neuro.ImageBasics import ResampleImages, SmoothImages
from nilearn import image
from nibabel.nifti1 import Nifti1Image
import numpy as np


class NeuroTest(unittest.TestCase):

    def setUp(self):
        self.test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/')
        self.atlas_name = "AAL"
        self.X = AtlasLibrary().get_nii_files_from_folder(self.test_folder, extension=".nii")

    def tearDown(self):
        pass

    def test_brain_atlas_load(self):

        brain_atlas = AtlasLibrary().get_atlas(self.atlas_name)

        # manually load brain atlas
        man_map = image.load_img(os.path.dirname(inspect.getfile(BrainAtlas)) + '/Atlases/AAL_SPM12/AAL.nii.gz').get_data()
        self.assertTrue(np.array_equal(man_map, brain_atlas.map))

    def test_brain_atlas(self):


        # 4101
        # Hippocampus_L
        # 4102
        # Hippocampus_R
        # 4201
        # Amygdala_L
        # 4202
        # Amygdala_R

        brain_atlas = BrainAtlas("AAL", "vec", rois=["Hippocampus_R", "Hippocampus_L", "Amygdala_L", "Amygdala_R"])
        new_data = brain_atlas.transform(self.X)
        self.assertTrue(len(self.X), len(brain_atlas.rois))

        brain_atlas_mean = BrainAtlas("AAL", "mean", rois='all')
        brain_atlas_mean.transform(self.X)
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
            nmb3.add(PhotonBatchElement(neuro_class_str, batch_size=5, **param_dict))
            instance_list.append(nmb3)

            nmb4 = NeuroModuleBranch(name="batched multi core application", nr_of_processes=3)
            nmb4.add(PhotonBatchElement(neuro_class_str, batch_size=5, **param_dict))
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






