"""
===========================================================
Project: PHOTON EXAMPLES
===========================================================
Description
-----------
Sample Pairing example

Version
-------
Created:        DD-MM-YYYY
Last updated:   DD-MM-YYYY


Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""


from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from photonai.optimization.Hyperparameters import Categorical
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston, load_breast_cancer
import numpy as np


# # WE USE THE BREAST CANCER SET FROM SKLEARN
# X, y = load_boston(True)
#
# # DESIGN YOUR PIPELINE
# my_pipe = Hyperpipe('sample_pairing_example',  # the name of your pipeline
#                     optimizer='grid_search',  # which optimizer PHOTON shall use
#                     metrics=['mean_squared_error', 'pearson_correlation'],  # the performance metrics of your interest
#                     best_config_metric='mean_squared_error',  # after hyperparameter search, the metric declares the winner config
#                     outer_cv=KFold(n_splits=5),  # repeat hyperparameter search three times
#                     inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively
#                     verbosity=1) # get error, warn and info message                    )
#
#
# # ADD ELEMENTS TO YOUR PIPELINE
# # first normalize all features
# my_pipe += PipelineElement('StandardScaler')
#
# # add sample pairing
# my_pipe += PipelineElement('SamplePairingRegression', {'draw_limit': [500, 1000, 10000],
#                                                        'generator': Categorical(['nearest_pair', 'random_pair'])},
#                            distance_metric='euclidean', test_disabled=True)
#
# # engage and optimize the good old SVM for Classification
# my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': [10]})
#
# # NOW TRAIN YOUR PIPELINE
# my_pipe.fit(X, y)



# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('sample_pairing_example_classification',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=5),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively
                    verbosity=1) # get error, warn and info message                    )


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# add sample pairing
my_pipe += PipelineElement('SamplePairingClassification', {'draw_limit': [500, 1000, 10000],
                                                       'generator': Categorical(['nearest_pair', 'random_pair'])},
                           distance_metric='euclidean', test_disabled=True)

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('RandomForestClassifier', hyperparameters={'n_estimators': [10]})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)



debug = True


