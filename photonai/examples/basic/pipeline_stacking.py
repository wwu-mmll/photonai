from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineStack
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)


my_pipe = Hyperpipe('basic_stacking',
                    optimizer='sk_opt',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1)

my_pipe += PipelineElement('StandardScaler')
my_pipe_stack = PipelineStack('final_stack', voting=False)
my_pipe_stack += PipelineElement('DecisionTreeClassifier', hyperparameters={'criterion': ['gini'],
                                                                            'min_samples_split': IntegerRange(2, 4)})

my_pipe_stack += PipelineElement('LinearSVC', hyperparameters={'C': FloatRange(0.5, 25)})
my_pipe += my_pipe_stack

my_pipe += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear'])})
my_pipe.fit(X, y)

debug = True
