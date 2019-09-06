from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import FloatRange, Categorical
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


# GET DATA
X, y = load_breast_cancer(True)

# CREATE HYPERPIPE
my_pipe = Hyperpipe('basic_switch_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1)

# Transformer Switch
my_pipe += Switch('TransformerSwitch', [PipelineElement('StandardScaler'), PipelineElement('PCA', test_disabled=True)])

# Estimator Switch
svm = PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']), 'C': FloatRange(0.5, 2, "linspace", num=5)})
tree = PipelineElement('DecisionTreeClassifier')

my_pipe += Switch('EstimatorSwitch', [svm, tree])

# FIT PIPELINE
my_pipe.fit(X, y)

debug = True
