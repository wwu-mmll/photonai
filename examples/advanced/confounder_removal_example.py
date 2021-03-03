from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement

# WE USE THE BREAST CANCER SET FROM SKLEARN
data = load_breast_cancer()
y = data.target

# now let's assume we want to regress out the effect of mean_radius and mean_texture
X = data.data[:, 2:]
mean_radius = data.data[:, 0]
mean_texture = data.data[:, 1]

# BUILD HYPERPIPE
pipe = Hyperpipe('confounder_removal_example',
                 optimizer='grid_search',
                 metrics=['accuracy', 'precision', 'recall'],
                 best_config_metric='accuracy',
                 outer_cv=KFold(n_splits=5),
                 inner_cv=KFold(n_splits=3),
                 verbosity=1,
                 project_folder='./tmp/')

# # there are two ways of specifying multiple confounders
# # first, you can simply pass a dictionary with "confounder" as key and a data matrix or list as value
# pipe += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=False)
# pipe.fit(X, y, confounder=[mean_radius, mean_texture])
# pipe += PipelineElement('SVC')

# second, you can also specify the names of the variables that should be used in the confounder removal step
pipe += PipelineElement('ConfounderRemoval', {},
                        standardize_covariates=True,
                        test_disabled=True,
                        confounder_names=['mean_radius', 'mean_texture'])

pipe += PipelineElement('SVC')

# those names must be keys in the kwargs dictionary
pipe.fit(X, y,  mean_radius=mean_radius, mean_texture=mean_texture)
