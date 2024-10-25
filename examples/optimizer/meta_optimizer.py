from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai import Hyperpipe, PipelineElement, Switch, FloatRange, IntegerRange

my_pipe = Hyperpipe('hp_switch_optimizer',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    optimizer='switch',
                    # optimizer_params={'name': 'grid_search'},
                    optimizer_params={'name': 'random_search', 'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    project_folder='./tmp',
                    verbosity=1)

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': IntegerRange(10, 30, step=5)},
                           test_disabled=True)

# set up two learning algorithms in an ensemble
estimator_selection = Switch('estimators')

estimator_selection += PipelineElement('RandomForestClassifier',
                                       criterion='gini',
                                       hyperparameters={'min_samples_split': IntegerRange(2, 4),
                                                        'max_features': ['sqrt', 'log2'],
                                                        'bootstrap': [True, False]})
estimator_selection += PipelineElement('SVC',
                                       hyperparameters={'C': FloatRange(0.5, 25, num=10),
                                                        'kernel': ['linear', 'rbf']})

my_pipe += estimator_selection

X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)

print(my_pipe.results_handler.get_mean_of_best_validation_configs_per_estimator())
