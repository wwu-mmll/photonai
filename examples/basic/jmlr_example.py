from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Stack
from photonai.optimization import FloatRange, IntegerRange

X, y = load_breast_cancer(return_X_y=True)


my_pipe = Hyperpipe('example_project',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3))

my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components':
                                            FloatRange(0.5, 0.8, step=0.1)},
                           test_disabled=True)

my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name':
                                            ['RandomUnderSampler','SMOTE']},
                           test_disabled=True)

# set up two learning algorithms in an ensemble
ensemble_learner = Stack('estimators', use_probabilities=True)
ensemble_learner += PipelineElement('DecisionTreeClassifier',
                                    criterion='gini',
                                    hyperparameters={'min_samples_split':
                                                     IntegerRange(2, 4)})
ensemble_learner += PipelineElement('LinearSVC',
                                    hyperparameters={'C':
                                                     FloatRange(0.5, 25)})

my_pipe += ensemble_learner
my_pipe += PipelineElement('SVC', kernel='rbf')
my_pipe.fit(X, y)

