from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro.AtlasMapping import AtlasMapper
from photonai.neuro.NeuroBase import NeuroModuleBranch
from sklearn.model_selection import KFold
from nilearn.datasets import fetch_oasis_vbm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# GET DATA FROM OASIS
n_subjects = 50
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(save_predictions='best')


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error', 'mean_squared_error'],  # the performance metrics of your interest
                    best_config_metric='mean_absolute_error',
                    inner_cv=KFold(n_splits=2),  # test each configuration ten times respectively,
                    verbosity=2,
                    output_settings=mongo_settings,
                    cache_folder='/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test_OASIS/cache')  # get error, warn and info message


brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
                                rois=['Network_1', 'Network_2'])

neuro_branch = NeuroModuleBranch('NeuroBranch')
neuro_branch += brain_atlas

my_pipe += PipelineElement('SVR', hyperparameters={}, kernel='linear')

# NOW TRAIN YOUR PIPELINE
my_folder = '/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test_OASIS'
atlas_mapper = AtlasMapper()
atlas_mapper.generate_mappings(my_pipe, neuro_branch, my_folder)
atlas_mapper.fit(X, y)

# LOAD TRAINED ATLAS MAPPER AND PREDICT
atlas_mapper = AtlasMapper()
atlas_mapper.load_from_file('/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test_oasis/basic_svm_pipe_atlas_mapper_meta.json')

print(atlas_mapper.predict(X))
debug = True


