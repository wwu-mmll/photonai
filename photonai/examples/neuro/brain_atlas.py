import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Stack
from photonai.neuro import NeuroBranch
from photonai.neuro.brain_atlas import AtlasLibrary

warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)



# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Limbic_System',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 verbosity=2,
                 cache_folder="./cache",
                 eval_final_performance=False,
                 output_settings=OutputSettings(project_folder='./tmp/'))

"""
AVAILABLE ATLASES
    'AAL'
    'HarvardOxford_Cortical_Threshold_25'
    'HarvardOxford_Subcortical_Threshold_25'
    'HarvardOxford_Cortical_Threshold_50'
    'HarvardOxford_Subcortical_Threshold_50'
    'Yeo_7'
    'Yeo_7_Liberal'
    'Yeo_17'
    'Yeo_17_Liberal'
    'Schaefer2018_*Parcels_*Networks' (replace first asterisk with 100, 200, ..., 1000 and second with 7 or 17)
"""
# to list all roi names of a specific atlas, you can do the following
AtlasLibrary().list_rois('AAL')
AtlasLibrary().list_rois('HarvardOxford_Cortical_Threshold_25')
AtlasLibrary().list_rois('HarvardOxford_Subcortical_Threshold_25')
AtlasLibrary().list_rois('Schaefer2018_100Parcels_7Networks')

# PICK AN ATLAS
# V1.1 ----------------------------------------------------------------
atlas = PipelineElement('BrainAtlas',
                        rois=['Hippocampus_L', 'Hippocampus_R', 'Amygdala_L', 'Amygdala_R'],
                        atlas_name="AAL", extract_mode='vec', batch_size=20)


neuro_branch_v1 = NeuroBranch('NeuroBranch', nr_of_processes=3)
neuro_branch_v1 += atlas

# V1.2 ----------------------------------------------------------------
atlas = PipelineElement('BrainAtlas',
                        rois=['all'],
                        atlas_name="Schaefer2018_100Parcels_7Networks", extract_mode='vec', batch_size=20)


neuro_branch_v2 = NeuroBranch('NeuroBranch', nr_of_processes=3)
neuro_branch_v2 += atlas

# V2 -------------------------------------------------------------
# it's also possible to combine ROIs from different atlases
neuro_stack = Stack('HarvardOxford')

ho_sub = NeuroBranch('HO_Subcortical')
ho_sub += PipelineElement('BrainAtlas',
                          rois=['Left Thalamus', 'Left Caudate', 'Left Putamen', 'Left Pallidum'],
                          atlas_name="HarvardOxford_Subcortical_Threshold_25", extract_mode='vec', batch_size=20)

ho_cort = NeuroBranch('HO_Cortical')
ho_cort += PipelineElement('BrainAtlas',
                           rois=['Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus'],
                           atlas_name="HarvardOxford_Cortical_Threshold_25", extract_mode='vec', batch_size=20)

neuro_stack += ho_cort
neuro_stack += ho_sub

# ADD NEURO ELEMENTS TO HYPERPIPE
# V1.1 --------------------------------------------
#pipe += neuro_branch_v1
# V1.2 --------------------------------------------
pipe += neuro_branch_v2
# V2 --------------------------------------------
# pipe += neuro_stack
# ------------------------------------------------
pipe += PipelineElement('PCA', n_components=20)
pipe += PipelineElement('RandomForestRegressor')

pipe.fit(X, y)
