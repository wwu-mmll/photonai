import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import FloatRange, IntegerRange
from photonai.photonlogger import logger

# setup training and test workflow
my_pipe = Hyperpipe('heart_failure_v1',
                    outer_cv=ShuffleSplit(n_splits=2, test_size=0.2),
                    inner_cv=KFold(n_splits=5, shuffle=True),
                    metrics=['balanced_accuracy', 'f1_score', 'matthews_corrcoef',
                             'sensitivity', 'specificity'],
                    best_config_metric='matthews_corrcoef',
                    optimizer='switch',
                    optimizer_params={'name': 'sk_opt', 'n_configurations': 5},
                    project_folder='./tmp',
                    verbosity=0)


# arrange a sequence of algorithms subsequently applied
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SimpleImputer')

# my_pipe += PipelineElement('PCA',
#                            hyperparameters={'n_components':
#                                             FloatRange(0.3, 0.9, step=0.1)},
#                            test_disabled=True)

# my_pipe += PipelineElement('LassoFeatureSelection',
#                            hyperparameters={'percentile': FloatRange(0.05, 0.4),
#                                             'alpha': FloatRange(0.1, 2, range_type="logspace")})
#
# my_pipe += PipelineElement('ImbalancedDataTransformer',
#                            hyperparameters={'method_name':
#                                             Categorical(['RandomUnderSampler',
#                                                          'RandomOverSampler',
#                                                          'SMOTE'])},
#                            test_disabled=True)

# compare different learning algorithms in an OR_Element
estimators = Switch('estimator_selection')

estimators += PipelineElement('RandomForestClassifier', criterion='gini', bootstrap=True,
                              hyperparameters={'min_samples_split': IntegerRange(2, 4),
                                               'max_features': ['auto', 'sqrt', 'log2']})

estimators += PipelineElement('GradientBoostingClassifier',
                              hyperparameters={'loss': ['deviance', 'exponential'],
                                               'learning_rate': FloatRange(0.001, 1,
                                                                           "logspace")})
estimators += PipelineElement('SVC',
                              hyperparameters={'C': FloatRange(0.5, 25),
                                               'kernel': ['linear', 'rbf']})

estimators += PipelineElement('MLPClassifier', learning_rate='adaptive',
                              hyperparameters={'activation': ['logistic', 'tanh', 'relu'],
                                               'alpha': FloatRange(0.001, 1, "logspace")})

estimators += PipelineElement('GaussianNB', priors=(0.7, 0.3))

my_pipe += estimators

# read data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# start the training, optimization and test procedure
my_pipe.fit(X, y)

my_pipe.results_handler.get_mean_of_best_validation_configs_per_estimator()
best_configs_dict = my_pipe.results_handler.get_n_best_validation_configs_per_estimator(estimator_names=["SVC",
                                                                                                         "GradientBoostingClassifier",
                                                                                                         "RandomForestClassifier"])
debug = True