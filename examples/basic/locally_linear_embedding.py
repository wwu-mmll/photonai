import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(name='llembedding_pipe',
                    optimizer='grid_search',
                    metrics=['balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    inner_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    project_folder='./tmp/',
                    verbosity=0)

my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('LocallyLinearEmbedding', n_components=math.ceil(0.1 * len(X[0])))

my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(1, 6)},
                           gamma='scale')

my_pipe.fit(X, y)
