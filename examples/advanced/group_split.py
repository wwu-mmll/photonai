import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

groups = np.random.random_integers(0, 3, (len(y), ))

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('group_split_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=GroupKFold(n_splits=4),
                    inner_cv=GroupShuffleSplit(n_splits=10),
                    verbosity=1,
                    project_folder='./tmp/')

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': [5, 10, None]}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y, groups=groups)
