from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Switch, OutputSettings
from photonai.optimization import FloatRange, Categorical

# GET DATA
X, y = load_breast_cancer(True)

# CREATE HYPERPIPE
settings = OutputSettings(project_folder='./tmp/')
my_pipe = Hyperpipe('basic_switch_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    output_settings=settings)

# Transformer Switch
my_pipe += Switch('TransformerSwitch', [PipelineElement('StandardScaler'), PipelineElement('PCA', test_disabled=True)])

# Estimator Switch
svm = PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']), 'C': FloatRange(0.5, 2, "linspace", num=5)})
tree = PipelineElement('DecisionTreeClassifier')

my_pipe += Switch('EstimatorSwitch', [svm, tree])

# FIT PIPELINE
my_pipe.fit(X, y)

debug = True
