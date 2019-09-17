from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import FloatRange, Categorical, IntegerRange
from photonai.neuro import NeuroBranch
from sklearn.model_selection import KFold
from nilearn.datasets import fetch_oasis_vbm
import numpy as np
import warnings
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
folder = './oasis_age'
settings = OutputSettings(project_folder=folder)

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Limbic_System',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=KFold(n_splits=2),
                 inner_cv=KFold(n_splits=3, shuffle=True),
                 output_settings=settings,
                 verbosity=1,
                 cache_folder=folder + "/cache4",
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

