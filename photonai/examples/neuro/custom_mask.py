import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro import NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# DESIGN YOUR PIPELINE
pipe = Hyperpipe('CustomMask',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 verbosity=2,
                 cache_folder="./cache",
                 eval_final_performance=False,
                 output_settings=OutputSettings(project_folder='./tmp/'))

local_spm_installation = '/spm-data/it-share/software_metis/SPMLauncher/ManagedSoftware/spm/'
custom_mask = local_spm_installation + 'spm12/toolbox/Anatomy/PMaps/Insula_Ig1.nii'
mask = PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec', batch_size=20)

# EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
# we recommend to always use neuro elements within a branch
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += mask

pipe += neuro_branch

pipe += PipelineElement('PCA', n_components=10)
pipe += PipelineElement('RandomForestRegressor')

pipe.fit(X, y)
