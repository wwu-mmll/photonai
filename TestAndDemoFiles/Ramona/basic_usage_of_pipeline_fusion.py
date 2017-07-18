import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch, PipelineStacking
from sklearn.model_selection import KFold

""" MORE DOCUMENTATION CAN BE FOUND HERE:
 https://translap.atlassian.net/wiki/display/PENIS/Photon+Toolbox+Framework+HowTos
"""

# Load data
surface = DataContainer()
# load ENIGMA surface values
surface += Features('../EnigmaTestFiles/CorticalMeasuresENIGMA_SurfAvg.csv',
                    usecols=np.arange(1, 73), na_values='NA')

# add sex as target
surface += Targets('../EnigmaTestFiles/Covariates.csv', usecols=[4],
                   na_values='NA')

# add age as covariate
surface += Covariates('age', '../EnigmaTestFiles/Covariates.csv',
                      usecols=[3], na_values='NA')

# add values from another file, namely ENIGMA thickness values, to features
thickness = DataContainer()
thickness += Features('../EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv',
                      usecols=np.arange(1, 73), na_values='NA')
# we have the same target as above
thickness.targets = surface.targets

# print(np.count_nonzero(np.isnan(surface.features.values)))
# print(np.count_nonzero(np.isnan(thickness.features.values)))

# we always do n_splits=3
cv_object = KFold(n_splits=3)

# make a global pipeline
manager = Hyperpipe('god', cv_object, metrics=['accuracy', 'precision'],
                    optimizer='timeboxed_random_grid_search',
                    optimizer_params={'limit_in_minutes': 1})

# we use the same optimizer for all pipelines = global hyperparameter search
global_optimizer = manager.optimizer

# REMARK: If we want to inject data apart from the sklearn pipeline forwarding of data we can use overwrite_x and _y

# make surface pipeline
surface_pipe = Hyperpipe('surface', cv_object, optimizer=global_optimizer, local_search=False,
                         overwrite_x=surface.features.values,
                         overwrite_y=np.ravel(surface.targets.values))
surface_pipe += PipelineElement.create('pca', {'n_components': np.arange(10, 70, 10).tolist()}, test_disabled=True)
surface_pipe += PipelineElement.create('svc', {'C': [1, 2]}, kernel='rbf')

# make thickness pipeline
thickness_pipe = Hyperpipe('thickness', cv_object, optimizer=global_optimizer, local_search=False,
                           overwrite_x=thickness.features.values,
                           overwrite_y=np.ravel(thickness.targets.values))
thickness_pipe += PipelineElement.create('pca', {'n_components': np.arange(10, 70, 10).tolist()}, test_disabled=True)
thickness_pipe += PipelineElement.create('svc', {'C': [1, 2]}, kernel='rbf')

# in the end we want to join both predictions as a new feature set
feature_union = PipelineStacking('surface_and_thickness', [surface_pipe, thickness_pipe])

# add the container for both surface and thickness pipe to the global pipe
manager += feature_union

# # use either SVM or Logistic regression
svc_estimator = PipelineElement.create('logistic', {'C': np.arange(0.2, 1, 0.2)})
manager += svc_estimator

# optimizes hyperparameters
X = surface.features.values
y = np.ravel(surface.targets.values)
manager.fit(X, y)
