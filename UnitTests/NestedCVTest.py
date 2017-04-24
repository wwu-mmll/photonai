import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch, PipelineFusion
from sklearn.model_selection import KFold


# Load data
data_object = DataContainer()

# create dummy data consisting of only indices
indices = np.reshape(np.arange(0, 300),(300, 1))
targets = np.random.randint(0, 2, (300))
data_object += Features(indices)
data_object += Targets(targets)


# example hyperparameter optimization for pipeline:
cv_object = KFold(n_splits=3)

manager = Hyperpipe('god', cv_object, optimizer='grid_search')

# we use the same optimizer for all pipelines = global hyperparameter search
global_optimizer = manager.optimizer

# make surface pipeline
pipe_1 = Hyperpipe('pipe1', cv_object, optimizer='grid_search', local_search=True)
pipe_1 += PipelineElement.create('pca', {'n_components': [1]}, set_disabled=True)
pipe_1 += PipelineElement.create('svc', {'C': [1, 2]}, kernel='rbf')

pipe_2 = Hyperpipe('pipe2', cv_object, optimizer='grid_search', local_search=True)
pipe_2 += PipelineElement.create('svc', {'C': [1, 2]}, kernel='rbf')

feature_union = PipelineFusion('nested_pipe', [pipe_1, pipe_2])

# add the container for both surface and thickness pipe to the global pipe
manager += feature_union

svc_estimator = PipelineElement.create('svc', {'C': [0.5, 1], 'kernel': ['rbf']})

manager += svc_estimator

# optimizes hyperparameters
X = data_object.features.values
y = np.ravel(data_object.targets.values)
manager.fit(X, y)

