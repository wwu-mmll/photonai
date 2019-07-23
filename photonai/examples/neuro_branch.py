
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.NeuroBase import NeuroModuleBranch

from sklearn.model_selection import KFold
import time
import os
import pandas as pd


file = '/home/rleenings/Projects/TestNeuro/PAC2018_age.csv'
data_folder = '/spm-data/Scratch/spielwiese_ramona/PAC2018/data_all/'

df = pd.read_csv(file)
X = [os.path.join(data_folder, f) + ".nii" for f in df["PAC_ID"]]
y = df["Age"]


X = X[:10]
y = y[:10]

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('amygdala_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error'],  # the performance metrics of your interest
                    best_config_metric='mean_absolute_error',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively,
                    verbosity=1)  # get error, warn and info message

preprocessing = PreprocessingPipe()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing


neuro_branch = NeuroModuleBranch('amygdala', nr_of_processes=3, cache_folder="/home/rleenings/Projects/TestNeuro/")

neuro_branch += PipelineElement('SmoothImages', hyperparameters={'fwhm': IntegerRange(3, 15)})
neuro_branch += PipelineElement('ResampleImages', hyperparameters={'voxel_size': IntegerRange(1, 5)})
neuro_branch += PipelineElement('BrainAtlas', hyperparameters={'rois': ['Hippocampus_L', 'Hippocampus_R',
                                                                        'Amygdala_L', 'Amygdala_R']},
                                atlas_name="AAL", extract_mode='vec')


# neuro_branch.test_transform(X, 3, '/home/rleenings/Projects/TestNeuro/', **{'BrainAtlas__rois': ['Amygdala_L'],
#                                                                           'SmoothImages__fwhm': [10, 10, 10],
#                                                                           'ResampleImages__voxel_size': [3, 3, 3]})

my_pipe += neuro_branch

my_pipe.add(PipelineElement('StandardScaler'))
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': None}, test_disabled=True)
my_pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2)}, gamma='scale')


# NOW TRAIN YOUR PIPELINE
start_time = time.time()
my_pipe.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


debug = True


