from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.neuro.AtlasMapping import AtlasMapper
from photonai.neuro.NeuroBase import NeuroModuleBranch
from sklearn.model_selection import KFold
from nilearn.datasets import fetch_oasis_vbm
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Specify where the results should be written to and the name of your analysis
results_folder = '/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test_oasis'
cache_folder = '/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test_oasis/cache'
analysis_name = 'atlas_mapping'

# GET DATA FROM OASIS
n_subjects = 200
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
y = np.array(age)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(save_predictions='best')


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(analysis_name,
                    optimizer='grid_search',
                    metrics=['mean_absolute_error', 'mean_squared_error'],
                    best_config_metric='mean_absolute_error',
                    inner_cv=KFold(n_splits=2),
                    verbosity=2,
                    output_settings=mongo_settings,
                    cache_folder=cache_folder)


brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
                                rois='all', batch_size=200)
#brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL", rois='all', batch_size=200)

neuro_branch = NeuroModuleBranch('NeuroBranch')
neuro_branch += brain_atlas

my_pipe += PipelineElement('LinearSVR')

# NOW TRAIN YOUR PIPELINE
atlas_mapper = AtlasMapper()
atlas_mapper.generate_mappings(my_pipe, neuro_branch, results_folder)
atlas_mapper.fit(X, y)

# LOAD TRAINED ATLAS MAPPER AND PREDICT
atlas_mapper = AtlasMapper()
# you can either load an atlas mapper by specifying the atlas_mapper_meta.json file that has been created during fit()
# or simply specify the results folder in which your model was saved (and you can also specify the analysis name in case
# there are multiple atlas mapper within one folder)

#atlas_mapper.load_from_file(os.path.join(results_folder) + 'atlas_mapper_meta.json')
atlas_mapper.load_from_folder(folder=results_folder, analysis_name=analysis_name)
print(atlas_mapper.predict(X))
debug = True


