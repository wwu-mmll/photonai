import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, Switch, Stack, OutputSettings
from photonai.optimization import FloatRange, IntegerRange, Categorical

# csv_data = pd.read_csv('~/Downloads/pd_speech_features.csv').to_numpy()
# X = csv_data[:, 1:-1]
# y = csv_data[:, -1]

# df = pd.read_csv('https://drive.google.com/uc?id=1kDE9mZaDFGfoIlgMbN9T2jaoF0056h9R')
# X = df.iloc[:, 1::]
# y = df.iloc[:, 0]

df = pd.read_csv('https://drive.google.com/uc?id=1V0eKMG0RwFkOz2EHDvbsdPmYqAd3f1pr')

X = df.iloc[:, 0:12]
y = df.iloc[:, 12]


my_pipe = Hyperpipe('example_project',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 50},
                    metrics=['balanced_accuracy', 'f1_score', 'matthews_corrcoef', 'accuracy', 'precision', 'recall'],
                    best_config_metric='matthews_corrcoef',
                    outer_cv=ShuffleSplit(n_splits=100, test_size=0.2),
                    inner_cv=KFold(n_splits=10, shuffle=True),
                    output_settings=OutputSettings(project_folder='./tmp'),
                    verbosity=1)

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SimpleImputer')
my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components':
                                            FloatRange(0.5, 0.8, step=0.1)},
                           test_disabled=True)

my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name':
                                            Categorical(['RandomUnderSampler', 'RandomOverSampler', 'SMOTE'])},
                           test_disabled=True)

# set up two learning algorithms in an ensemble
estimator_selection = Switch('estimators')
estimator_selection += PipelineElement('RandomForestClassifier',
                                       criterion='gini',
                                       hyperparameters={'min_samples_split': IntegerRange(2, 4),
                                                        'max_features': ['auto', 'sqrt', 'log2'],
                                                        'bootstrap': [True, False]})
estimator_selection += PipelineElement('SVC',
                                       hyperparameters={'C': FloatRange(0.5, 25),
                                                        'kernel': ['linear', 'rbf']})

estimator_selection += PipelineElement('MLPClassifier',
                                       learning_rate='adaptive',
                                       hyperparameters={'activation': ['logistic', 'tanh', 'relu'],
                                                        'alpha': FloatRange(0.001, 1, "geomspace")})

estimator_selection += PipelineElement('GradientBoostingClassifier',
                                       hyperparameters={'loss': ['deviance', 'exponential'],
                                                        'learning_rate': FloatRange(0.001, 1, "geomspace")})

my_pipe += estimator_selection
my_pipe.fit(X, y)

