import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch, PipelineFusion
from sklearn.model_selection import KFold

# LOAD DATA
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# SET UP HYPERPIPE
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                    metrics=['accuracy', 'precision'],
                    hyperparameter_specific_config_cv_object=KFold(n_splits=3),
                    hyperparameter_search_cv_object=KFold(n_splits=3))

my_pipe += PipelineElement.create('standard_scaler', test_disabled=True)
my_pipe.add(PipelineElement.create('pca', {'n_components': [1, 2]}, test_disabled=True, whiten=False))
my_pipe += PipelineElement.create('svc', {'C': [0.5, 1], 'kernel': ['rbf', 'linear']})

# START HYPERPARAMETER SEARCH
my_pipe.fit(X, y)

# AFTER FINDING THE BEST PARAMETER COMBINATION
# you can access the optimum pipeline via "optimum_pipe":
# and use it to predict new data by:
# new_data = []
# my_pipe.optimum_pipe.predict(new_data)

