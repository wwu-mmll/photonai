from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical

# loading the iris dataset
X, y = load_iris(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('multi_class_svm_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=3, shuffle=True),
                    verbosity=1,
                    project_folder='./tmp/')


my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(0.5, 2)},
                           gamma='scale')

my_pipe.fit(X, y)
