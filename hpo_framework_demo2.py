import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch, PipelineFusion
from sklearn.model_selection import KFold

""" MORE DOCUMENTATION CAN BE FOUND HERE:
 https://translap.atlassian.net/wiki/display/PENIS/Photon+Toolbox+Framework+HowTos
"""

# Load data
surface = DataContainer()
# load ENIGMA surface values
surface += Features('EnigmaTestFiles/CorticalMeasuresENIGMA_SurfAvg.csv',
                    usecols=np.arange(1, 73), na_values='NA')

# add sex as target
surface += Targets('EnigmaTestFiles/Covariates.csv', usecols=[4],
                   na_values='NA')

# add age as covariate
surface += Covariates('age', 'EnigmaTestFiles/Covariates.csv',
                      usecols=[3], na_values='NA')

# add values from another file, namely ENIGMA thickness values, to features
thickness = DataContainer()
thickness += Features('EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv',
                      usecols=np.arange(1, 73), na_values='NA')
# we have the same target as above
thickness.targets = surface.targets

# we always do n_splits=3
cv_object = KFold(n_splits=3)

# make a global pipeline
manager = Hyperpipe('god', cv_object)

# we use the same optimizer for all pipelines = global hyperparameter search
global_optimizer = manager.optimizer

# make surface pipeline
surface_pipe = Hyperpipe('surface', cv_object, optimizer=global_optimizer, local_search=False, X=surface.features.values, y=np.ravel(surface.targets.values))
surface_pipe += PipelineElement.create('pca', {'n_components': np.arange(10, 70, 10).tolist()}, set_disabled=True)

# make thickness pipeline
thickness_pipe = Hyperpipe('thickness', cv_object, optimizer=global_optimizer, local_search=False, X=thickness.features.values, y=np.ravel(thickness.targets.values))
thickness_pipe += PipelineElement.create('pca', {'n_components': np.arange(10, 70, 10).tolist()}, set_disabled=True)

feature_union = PipelineFusion('surface_and_thickness', [surface_pipe, thickness_pipe])

# add the container for both surface and thickness pipe to the global pipe
manager += feature_union

# add a pca analysis, specify hyperparameters to test
manager += PipelineElement.create('pca', {'n_components': [1, 5, 10, 20]}, set_disabled=True)

# use either SVM or Logistic regression
svc_estimator = PipelineElement.create('svc', {'C': np.arange(0.2, 1, 0.2), 'kernel': ['rbf', 'sigmoid']})
# or Logistic regression
lr_estimator = PipelineElement.create('logistic', {'C': np.logspace(-4, 4, 5)})

# use PipelineSwitch Element to interchange the estimators
manager += PipelineSwitch('final_estimator', [svc_estimator, lr_estimator])

# optimizes hyperparameters
X = surface.features.values
y = np.ravel(surface.targets.values)
manager.fit(X, y)
