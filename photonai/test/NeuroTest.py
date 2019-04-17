import unittest, os, inspect
from ..neuro.BrainAtlas import AtlasLibrary, BrainAtlas
from nilearn import image
import numpy as np


class NeuroTest(unittest.TestCase):

    def setUp(self):
        self.test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/')
        self.atlas_name = "AAL"

    def tearDown(self):
        pass

    def test_brain_atlas_load(self):

        brain_atlas = AtlasLibrary().get_atlas(self.atlas_name)

        # manually load brain atlas
        man_map = image.load_img(os.path.dirname(inspect.getfile(BrainAtlas)) + '/Atlases/AAL_SPM12/AAL.nii.gz').get_data()
        self.assertTrue(np.array_equal(man_map, brain_atlas.map))

    def test_brain_atlas(self):
        X = AtlasLibrary().get_nii_files_from_folder(self.test_folder, extension=".nii")

        # 4101
        # Hippocampus_L
        # 4102
        # Hippocampus_R
        # 4201
        # Amygdala_L
        # 4202
        # Amygdala_R

        brain_atlas = BrainAtlas("AAL", "vec", rois=["Hippocampus_R", "Hippocampus_L", "Amygdala_L", "Amygdala_R"])
        new_data = brain_atlas.transform(X)
        self.assertTrue(len(X), len(brain_atlas.rois))

        brain_atlas_mean = BrainAtlas("AAL", "mean", rois='all')
        brain_atlas_mean.transform(X)
        debug = True





