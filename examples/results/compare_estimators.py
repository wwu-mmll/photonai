from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_breast_cancer

from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import FloatRange, IntegerRange
from photonai.processing import ResultsHandler

X, y = load_breast_cancer(return_X_y=True)

# setup training and test workflow
my_pipe = Hyperpipe('compare_estimators',
                    outer_cv=ShuffleSplit(n_splits=5, test_size=0.2),
                    inner_cv=KFold(n_splits=5, shuffle=True),
                    metrics=['balanced_accuracy', 'f1_score', 'auc', 'matthews_corrcoef',
                             'accuracy', 'precision', 'recall'],
                    best_config_metric='balanced_accuracy',
                    optimizer='switch',
                    optimizer_params={'name': 'sk_opt', 'n_configurations': 10},
                    project_folder='./tmp',
                    verbosity=1)

# arrange a sequence of algorithms subsequently applied
my_pipe += PipelineElement('StandardScaler')

# compare different learning algorithms in an OR_Element
estimators = Switch('estimator_selection')

estimators += PipelineElement('RandomForestClassifier', criterion='gini', bootstrap=True,
                              hyperparameters={'min_samples_split': IntegerRange(2, 4),
                                               'max_features': ['auto', 'sqrt', 'log2']})

estimators += PipelineElement('GradientBoostingClassifier',
                              hyperparameters={'loss': ['deviance', 'exponential'],
                                               'learning_rate': FloatRange(0.001, 1, "logspace")})
estimators += PipelineElement('SVC',
                              hyperparameters={'C': FloatRange(0.5, 25),
                                               'kernel': ['linear', 'rbf']})

estimators += PipelineElement('GaussianNB')

my_pipe += estimators

# start the training, optimization and test procedure
my_pipe.fit(X, y)

my_pipe.results_handler.get_mean_of_best_validation_configs_per_estimator()

# or after training 
res = ResultsHandler()
res.load_from_file("./photon_result_file.json")
estimator_performances = res.get_mean_of_best_validation_configs_per_estimator()
