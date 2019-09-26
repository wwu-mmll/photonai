import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro import NeuroBranch
from photonai.neuro.brain_atlas import AtlasLibrary
from photonai.processing import ResultsHandler

warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# DEFINE OUTPUT SETTINGS
settings = OutputSettings(project_folder='./tmp/')

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
"""
# to list all roi names of a specific atlas, you can do the following
AtlasLibrary().list_rois('AAL')
AtlasLibrary().list_rois('HarvardOxford_Cortical_Threshold_25')
AtlasLibrary().list_rois('HarvardOxford_Subcortical_Threshold_25')

# PICK AN ATLAS
atlas = PipelineElement('BrainAtlas',
                        rois=['Hippocampus_L', 'Amygdala_L'],
                        atlas_name="AAL", extract_mode='vec', batch_size=20)

# EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += atlas


# ADD NEURO ELEMENTS TO HYPERPIPE

pipe += neuro_branch

pipe += PipelineElement('LinearSVR')

pipe.fit(X, y)

# GET IMPORTANCE SCORES
handler = ResultsHandler(pipe.results)

# get feature importances (training set) for your best configuration (for all outer folds)
# this function returns the importance scores for the best configuration of each outer fold in a list
importance_scores_outer_folds = handler.get_importance_scores()
importance_scores_optimum_pipe = handler.results.best_config_feature_importances

img, _, _ = pipe.optimum_pipe.inverse_transform(importance_scores_optimum_pipe, None)
img.to_filename('./tmp/best_config_feature_importances.nii.gz')
debug = True


