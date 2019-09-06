from photonai.base import Hyperpipe, PipelineElement, Stack
from photonai.optimization import FloatRange, IntegerRange, Categorical
from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)


my_pipe = Hyperpipe('basic_stacking',
                    optimizer='sk_opt',
                    optimizer_params={'num_iterations': 5},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1)

my_pipe += PipelineElement('StandardScaler')
my_pipe_stack = Stack('final_stack', voting=False)
my_pipe_stack += PipelineElement('DecisionTreeClassifier', hyperparameters={'criterion': ['gini'],
                                                                            'min_samples_split': IntegerRange(2, 4)})

my_pipe_stack += PipelineElement('LinearSVC', hyperparameters={'C': FloatRange(0.5, 25)})
my_pipe += my_pipe_stack

my_pipe += PipelineElement('LinearSVC')
my_pipe.fit(X, y)

debug = True
