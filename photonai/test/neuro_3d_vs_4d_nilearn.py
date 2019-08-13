
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe, CallbackElement
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.BrainAtlas import AtlasLibrary

from sklearn.model_selection import KFold
import time
import nilearn
import os
import pandas as pd


file = '/spm-data/Scratch/spielwiese_ramona/PAC2018/test/PAC2018_age.csv'
data_folder = '/spm-data/Scratch/spielwiese_ramona/PAC2018/data_all/'

df = pd.read_csv(file)
X = [os.path.join(data_folder, f) + ".nii" for f in df["PAC_ID"]]
y = df["Age"]


X = X[:50]
y = y[:50]

atlas_obj = AtlasLibrary().get_atlas('AAL', affine, shape, self.mask_threshold)
roi_objects = self._get_rois(atlas_obj, which_rois=self.rois, background_id=self.background_id)

masker = NiftiMasker(mask_img=self.mask_image.mask, target_affine=self.affine,
                                      target_shape=self.shape, dtype='float32')


