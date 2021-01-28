import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from photonai.base import Hyperpipe, PipelineElement, Switch, OutputSettings
from photonai.optimization import FloatRange, IntegerRange, Categorical, BestPerformanceConstraint, MinimumPerformanceConstraint

# setup training and test workflow
my_pipe = Hyperpipe('heart_failure_no_fu',
                    outer_cv=ShuffleSplit(n_splits=100, test_size=0.2),
                    inner_cv=KFold(n_splits=5, shuffle=True),
                    metrics=['balanced_accuracy', 'f1_score', 'auc', 'matthews_corrcoef',
                             'accuracy', 'precision', 'recall'],
                    best_config_metric='matthews_corrcoef',
                    optimizer='switch',
                    optimizer_params={'name': 'sk_opt', 'n_configurations': 50},
                    performance_constraints=[MinimumPerformanceConstraint('matthews_corrcoef', 0.35)],
                    output_settings=OutputSettings(project_folder='./tmp'),
                    verbosity=1)

# arrange a sequence of algorithms subsequently applied
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SimpleImputer')

# my_pipe += PipelineElement('PCA',
#                            hyperparameters={'n_components':
#                                             FloatRange(0.3, 0.9, step=0.1)},
#                            test_disabled=True)

my_pipe += PipelineElement('LassoFeatureSelection',
                           hyperparameters={'percentile_to_keep': FloatRange(0.1, 0.5),
                                            'alpha': FloatRange(0.1, 2, range_type="logspace")})

my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name':
                                            Categorical(['RandomUnderSampler',
                                                         'RandomOverSampler',
                                                         'SMOTE'])},
                           test_disabled=True)

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
df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
X = df.iloc[:, 0:11]
y = df.iloc[:, 12]

# start the training, optimization and test procedure
my_pipe.fit(X, y)

