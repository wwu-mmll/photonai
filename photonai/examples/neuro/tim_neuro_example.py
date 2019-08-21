from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PipelineStack, PipelineSwitch
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.BrainAtlas import AtlasLibrary
from photonai.base.PhotonBase import CallbackElement
from sklearn.model_selection import ShuffleSplit, KFold
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
folder = '/spm-data/Scratch/spielwiese_nils_winter/'
settings = OutputSettings(project_folder=folder)

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Limbic_System',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error'],
                 best_config_metric='mean_absolute_error',
                 #outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 #inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 outer_cv=KFold(n_splits=2),
                 inner_cv=KFold(n_splits=3, shuffle=True),
                 output_settings=settings,
                 verbosity=1,
                 cache_folder=folder + "/cache3",
                 eval_final_performance=True)

batch_size = 100
neuro_branch = NeuroModuleBranch('NeuroBranch') #, nr_of_processes=4)
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': Categorical([3, 5])}, batch_size=batch_size)
neuro_branch += CallbackElement('resample_monitor', my_monitor)
neuro_branch += PipelineElement('SmoothImages', {'fwhm': Categorical([6, 12])}, batch_size=batch_size)
neuro_branch += CallbackElement('smooth_monitor', my_monitor)
# neuro_branch += PipelineElement('BrainAtlas', rois=['Hippocampus_L', 'Hippocampus_R', 'Amygdala_L', 'Amygdala_R'], atlas_name="AAL", extract_mode='vec', batch_size=20)
neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter', extract_mode='vec', batch_size=batch_size)
neuro_branch += CallbackElement('mask_monitor', my_monitor)
#neuro_branch += PipelineElement('BrainMask', mask_image='MNI_ICBM152_WholeBrain', extract_mode='vec', batch_size=20)
pipe += neuro_branch
#pipe += CallbackElement('monitor_parallel_branch', my_monitor)
pipe += PipelineElement('PCA', n_components=20, test_disabled=False)
pipe += PipelineElement('SVR', kernel='linear')


pipe.fit(X, y)

debug = True
