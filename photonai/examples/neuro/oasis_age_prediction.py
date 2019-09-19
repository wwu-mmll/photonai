import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold
import os
from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro import NeuroBranch
from photonai.optimization import Categorical

warnings.filterwarnings("ignore", category=DeprecationWarning)

# GET DATA FROM OASIS
n_subjects = 100
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


def my_monitor(X, y=None, **kwargs):
    print(X.shape)
    debug = True


# DEFINE OUTPUT SETTINGS
base_folder = os.path.dirname(os.path.abspath(__file__))
cache_folder_path = os.path.join(base_folder, "cache")
tmp_folder_path = os.path.join(base_folder, "tmp")

settings = OutputSettings(project_folder=tmp_folder_path)

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Limbic_System',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=KFold(n_splits=2),
                 inner_cv=KFold(n_splits=3, shuffle=True),
                 output_settings=settings,
                 verbosity=1,
                 cache_folder=cache_folder_path,
                 eval_final_performance=True)

batch_size = 25
neuro_branch = NeuroBranch('NeuroBranch', nr_of_processes=4)
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': Categorical([3, 5])},
                                batch_size=batch_size)
neuro_branch += PipelineElement('SmoothImages', {'fwhm': Categorical([6, 12])},
                                batch_size=batch_size)
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec',
                                batch_size=batch_size)

pipe += neuro_branch

pipe += PipelineElement('PCA', n_components=20, test_disabled=False)
pipe += PipelineElement('SVR', kernel='linear')

pipe.fit(X, y)
