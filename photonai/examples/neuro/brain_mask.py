import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro import NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 10
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# DEFINE OUTPUT SETTINGS
settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('GrayMatter',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 verbosity=2,
                 cache_folder="./cache",
                 eval_final_performance=False,
                 output_settings=settings)

# CHOOSE BETWEEN MASKS
# available masks
# 'MNI_ICBM152_GrayMatter'
# 'MNI_ICBM152_WhiteMatter'
# 'MNI_ICBM152_WholeBrain'
# 'Cerebellum'

mask = PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter',
                          extract_mode='vec', batch_size=20)

# EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
# we recommend to always use neuro elements within a branch
neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += mask

pipe += neuro_branch
#pipe += mask

pipe += PipelineElement('PCA', n_components=5)

pipe += PipelineElement('RandomForestRegressor')

pipe.fit(X, y)
