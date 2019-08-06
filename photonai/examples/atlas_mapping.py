from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.AtlasMapping import AtlasMapper
from sklearn.model_selection import KFold
import time
import os
import pandas as pd
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#load nifti files
file_path_list = glob.glob('/spm-data/Scratch/spielwiese_vincent/PAC2019/TestRun_PreProcessing/mri/mwp1age*.nii')
print(file_path_list)
X = sorted(file_path_list)
print(X)

#load labels
PAClabels = pd.DataFrame(pd.read_excel(r'/spm-data/Scratch/spielwiese_ramona/PAC2019/old_files/PAC2019_data.xlsx'))
PAClabels = PAClabels[['id', 'age']]
PACIDs = [z.split("/mri/mwp1")[1].split("_")[0] for z in file_path_list]
SelectedLabels = PAClabels[PAClabels['id'].isin(PACIDs)]
y = SelectedLabels['age'].to_numpy()
print(np.isnan(y))
print(type(y))


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
                    cache_folder='/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test/cache')  # get error, warn and info message

preprocessing = PreprocessingPipe()
preprocessing += PipelineElement('BrainAtlas', atlas_name="AAL", extract_mode='vec', rois=['Amygdala_L', 'Amygdala_R'])
#preprocessing += PipelineElement('BrainAtlas', atlas_name="AAL", extract_mode='vec', rois='all')
#preprocessing += PipelineElement('BrainAtlas', atlas_name="AAL", extract_mode='vec', rois=['Amygdala_L', 'Amygdala_R',
#                                                                                           'Precentral_L', 'Precentral_R',
#                                                                                           'Frontal_Mid_L', 'Frontal_Mid_R'])

my_pipe += preprocessing
my_pipe += PipelineElement('SVR', hyperparameters={}, kernel='linear')


# NOW TRAIN YOUR PIPELINE
start_time = time.time()
atlas_mapper = AtlasMapper()
my_folder = '/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test'
atlas_mapper.generate_mappings(my_pipe, my_folder)
atlas_mapper.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# atlas_mapper = AtlasMapper()
# atlas_mapper.load_from_file('/spm-data/Scratch/spielwiese_nils_winter/atlas_mapper_test/basic_svm_pipe_atlas_mapper_meta.json')

print(atlas_mapper.predict(X))
debug = True


