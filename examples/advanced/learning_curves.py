from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange

my_pipe = Hyperpipe('basic_forest_pipe',
                    inner_cv=KFold(n_splits=2),
                    outer_cv=KFold(n_splits=2),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 15},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    project_folder='./tmp',
                    # this is how to make photonai calculate learning curves
                    # output and figures for this can be found in the project folder
                    learning_curves=True,
                    learning_curves_cut=FloatRange(0.1, 1, step=0.1))

my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('RandomForestClassifier')
X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)
