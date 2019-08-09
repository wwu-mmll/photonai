from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import Categorical, IntegerRange
from photonai.neuro.NeuroBase import NeuroModuleBranch

from sklearn.model_selection import KFold
from nilearn.datasets import fetch_oasis_vbm

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# DESIGN YOUR PIPELINE
settings = OutputSettings(project_folder='.')

my_pipe = Hyperpipe('amygdala_pipe',
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    cache_folder="./cache",
                    output_settings=settings)

preprocessing = PreprocessingPipe()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing

neuro_branch = NeuroModuleBranch('amygdala', nr_of_processes=1)
neuro_branch += PipelineElement('SmoothImages', {'fwhm': [3, 5, 10, 15]})
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': IntegerRange(1, 3)})
neuro_branch += PipelineElement('BrainAtlas', hyperparameters={'rois': [['Hippocampus_L', 'Hippocampus_R'],
                                                                        ['Amygdala_L', 'Amygdala_R']]},
                                atlas_name="AAL", extract_mode='vec')

my_pipe += neuro_branch

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['rbf', 'linear'])}, gamma='scale')

# NOW TRAIN YOUR PIPELINE
start_time = time.time()
my_pipe.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

debug = True


