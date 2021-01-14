import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from photonai.base import Hyperpipe, PipelineElement, Switch, OutputSettings
from photonai.optimization import FloatRange, IntegerRange, Categorical, BestPerformanceConstraint

# setup training and test workflow
my_pipe = Hyperpipe('example_project',
                    optimizer='switch',
                    optimizer_params={'name': 'sk_opt', 'n_configurations': 30},
                    metrics=['balanced_accuracy', 'f1_score', 'auc', 'matthews_corrcoef',
                             'accuracy', 'precision', 'recall'],
                    best_config_metric='matthews_corrcoef',
                    outer_cv=ShuffleSplit(n_splits=100, test_size=0.2),
                    inner_cv=KFold(n_splits=5, shuffle=True),
                    performance_constraints=[BestPerformanceConstraint('matthews_corrcoef')],
                    output_settings=OutputSettings(project_folder='./tmp'),
                    cache_folder='./cache',
                    verbosity=1)

# arrange a sequence of algorithms subsequently applied
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SimpleImputer')

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components':
                                            FloatRange(0.3, 0.9, step=0.1)},
                           test_disabled=True)

my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name':
                                            Categorical(['RandomUnderSampler',
                                                         'RandomOverSampler',
                                                         'SMOTE'])},
                           test_disabled=True)

# compare different learning algorithms in an OR_Element
estimators = Switch('estimator_selection')

estimators += PipelineElement('RandomForestClassifier', criterion='gini',
                              hyperparameters={'min_samples_split': IntegerRange(2, 4),
                                               'max_features': ['auto', 'sqrt', 'log2'],
                                               'bootstrap': [True, False]})

estimators += PipelineElement('GradientBoostingClassifier',
                              hyperparameters={'loss': ['deviance', 'exponential'],
                                               'learning_rate': FloatRange(0.001, 1,
                                                                           "geomspace")})
estimators += PipelineElement('SVC',
                              hyperparameters={'C': FloatRange(0.5, 25),
                                               'kernel': ['linear', 'rbf']})

estimators += PipelineElement('MLPClassifier', learning_rate='adaptive',
                              hyperparameters={'activation': ['logistic', 'tanh', 'relu'],
                                               'alpha': FloatRange(0.001, 1, "geomspace")})

estimators += PipelineElement('GaussianNB', priors=(0.7, 0.3))

my_pipe += estimators

# read data
df = pd.read_csv('https://drive.google.com/uc?id=1V0eKMG0RwFkOz2EHDvbsdPmYqAd3f1pr')
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# start the training, optimization and test procedure
my_pipe.fit(X, y)

