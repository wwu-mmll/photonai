from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.base.PhotonBatchElement import PhotonBatchElement
from photonai.neuro.NeuroBase import NeuroModuleBranch
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

file = '/spm-data/Scratch/spielwiese_ramona/PAC2018/PAC2018_age.csv'
data_folder = '/spm-data/Scratch/spielwiese_ramona/PAC2018/data_all/'

df = pd.read_csv(file)
X = [os.path.join(data_folder, f) + ".nii" for f in df["PAC_ID"]]
y = df["Age"]

n_subjects = 50

X = X[:n_subjects]
y = y[:n_subjects]


# DEFINE OUTPUT SETTINGS
settings = OutputSettings(project_folder='/spm-data/Scratch/spielwiese_nils_winter/brain_mask_test/')

# DESIGN YOUR PIPELINE
pipe = Hyperpipe('Amygdala',
                    optimizer='grid_search',
                    metrics=['mean_absolute_error'],
                    best_config_metric='mean_absolute_error',
                    outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                    verbosity=2,
                    cache_folder="/spm-data/Scratch/spielwiese_nils_winter/brain_atlas_test/cache",
                    eval_final_performance=False)

mask = PhotonBatchElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter',
                           extract_mode='vec', batch_size=20)

# EITHER ADD A NEURO BRANCH OR THE ATLAS ITSELF
neuro_branch = NeuroModuleBranch('NeuroBranch')
neuro_branch += mask

pipe += neuro_branch
#pipe += mask

pipe += PipelineElement('PCA', n_components=10)

pipe += PipelineElement('RandomForestRegressor')

pipe.fit(X, y)
