from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange

# GET DATA
X, y = load_breast_cancer(return_X_y=True)

# CREATE HYPERPIPE
my_pipe = Hyperpipe('basic_switch_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 15},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    project_folder='./tmp/')

# Transformer Switch
my_pipe += Switch('StandardizationSwitch',
                  [PipelineElement('StandardScaler'),
                   PipelineElement('MinMaxScaler')])

# Estimator Switch
svm = PipelineElement('SVC',
                      hyperparameters={'kernel': ['rbf', 'linear']})

tree = PipelineElement('DecisionTreeClassifier',
                       hyperparameters={'min_samples_split': IntegerRange(2, 5),
                                        'min_samples_leaf': IntegerRange(1, 5),
                                        'criterion': ['gini', 'entropy']})

my_pipe += Switch('EstimatorSwitch', [svm, tree])

my_pipe.fit(X, y)
