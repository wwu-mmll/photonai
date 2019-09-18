from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Stack, OutputSettings
from photonai.optimization import FloatRange, IntegerRange

X, y = load_breast_cancer(True)

settings = OutputSettings(project_folder='./tmp/')

my_pipe = Hyperpipe('basic_stack_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'num_iterations': 5},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)

my_pipe += PipelineElement('StandardScaler')

tree = PipelineElement('DecisionTreeClassifier', hyperparameters={'criterion': ['gini'],
                                                                            'min_samples_split': IntegerRange(2, 4)})

svc = PipelineElement('LinearSVC', hyperparameters={'C': FloatRange(0.5, 25)})

my_pipe += Stack('final_stack', [tree, svc])

my_pipe += PipelineElement('LinearSVC')
my_pipe.fit(X, y)

debug = True
