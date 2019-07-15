
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.AtlasMapping import AtlasMapper
from sklearn.model_selection import KFold
import time
import os
import pandas as pd

# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(save_predictions='best')



# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error'],  # the performance metrics of your interest
                    best_config_metric='mean_absolute_error',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively,
                    verbosity=1,
                    output_settings=mongo_settings)  # get error, warn and info message

preprocessing = PreprocessingPipe()
preprocessing += PipelineElement('BrainAtlas', atlas_name="AAL", extract_mode='vec')
my_pipe += preprocessing
my_pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2)}, gamma='scale')


# NOW TRAIN YOUR PIPELINE
start_time = time.time()
atlas_mapper = AtlasMapper()
my_folder = ''
atlas_mapper.generate_mappings(my_pipe, my_folder)
atlas_mapper.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


debug = True


