from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


# WE USE THE BREAST CANCER SET FROM SKLEARN
data = load_breast_cancer()
y = data.target

# now let's assume we want to regress out the effect of mean_radius and mean_texture
X = data.data[:, 2:]
mean_radius = data.data[:, 0]
mean_texture = data.data[:, 1]

# BUILD HYPERPIPE
pipe = Hyperpipe('basic_svm_pipe_no_performance',
                 optimizer='grid_search',
                 metrics=['accuracy', 'precision', 'recall'],
                 best_config_metric='accuracy',
                 outer_cv=KFold(n_splits=5),
                 inner_cv=KFold(n_splits=3),
                 verbosity=1)

pipe += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=False)

debug = True