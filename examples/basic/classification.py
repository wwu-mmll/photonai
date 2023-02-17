from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical, IntegerRange

my_pipe = Hyperpipe('basic_svm_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 15},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    project_folder='./tmp')

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': IntegerRange(10, 30)},
                           test_disabled=True)

my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(1, 6)},
                           gamma='scale')

X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)
