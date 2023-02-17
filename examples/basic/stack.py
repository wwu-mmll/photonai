from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Stack
from photonai.optimization import FloatRange, IntegerRange

X, y = load_breast_cancer(return_X_y=True)

my_pipe = Hyperpipe('basic_stack_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=0,
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')

tree = PipelineElement('DecisionTreeClassifier',
                       hyperparameters={'min_samples_split': IntegerRange(2, 4)},
                       criterion='gini')

svc = PipelineElement('LinearSVC', hyperparameters={'C': FloatRange(0.5, 25)})

# for a stack that includes estimators you can choose whether predict or predict_proba is called for all estimators
# in case only some implement predict_proba, predict is called for the remaining estimators
my_pipe += Stack('final_stack', [tree, svc], use_probabilities=True)

my_pipe += PipelineElement('LinearSVC')
my_pipe.fit(X, y)
