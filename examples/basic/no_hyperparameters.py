from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
from photonai import Hyperpipe, PipelineElement

my_pipe = Hyperpipe('basic_svm_pipe',
                    inner_cv=KFold(n_splits=5),
                    outer_cv=KFold(n_splits=3),
                    metrics=['balanced_accuracy',
                             'precision',
                             'recall',
                             'accuracy'],
                    best_config_metric='balanced_accuracy',
                    project_folder='./tmp')

my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA',
                           n_components=10,
                           test_disabled=True)

my_pipe += PipelineElement('LogisticRegression')


X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)

