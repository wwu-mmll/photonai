# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:03:03 2023

@author: Caroline Pinte
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from photonai import Hyperpipe, PipelineElement, IntegerRange, FloatRange

my_pipe = Hyperpipe('basic_regression_pipe',
                    optimizer='random_search',
                    optimizer_params={'n_configurations': 50},
                    metrics=['mean_squared_error', 'mean_absolute_error', 'explained_variance'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=10, shuffle=True),
                    inner_cv=KFold(n_splits=10, shuffle=True),
                    verbosity=0,
                    project_folder='./tmp/')

my_pipe += PipelineElement('SimpleImputer')

my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('LassoFeatureSelection',
                           hyperparameters={'percentile': [0.1, 0.2, 0.3],
                                            'alpha': FloatRange(0.5, 5)})

my_pipe += PipelineElement('RandomForestRegressor',
                           hyperparameters={'n_estimators': IntegerRange(10, 100)})

# load data and train
X, y = load_diabetes(return_X_y=True)
my_pipe.fit(X, y)