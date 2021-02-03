from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange

my_pipe = Hyperpipe('covid_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 15},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SimpleImputer')

svm = PipelineElement('SVC',
                      hyperparameters={'kernel': ['rbf', 'linear']})

tree = PipelineElement('DecisionTreeClassifier',
                       hyperparameters={'min_samples_split': IntegerRange(2, 5),
                                        'min_samples_leaf': IntegerRange(1, 5),
                                        'criterion': ['gini', 'entropy']})

my_pipe += Switch('EstimatorSwitch', [svm, tree])

X, y = None
my_pipe.fit(X, y)
