from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)

my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    write_to_db=True,
                    mongodb_connect_url="mongodb://localhost:27017/photon_db",
                    verbose=1,
                    save_all_predictions=False)


svm = PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

tree = PipelineElement('DecisionTreeClassifier',  {'criterion': Categorical(['gini', 'entropy']),
                                                   'min_samples_split': IntegerRange(2, 5)})

switch = PipelineSwitch('estimator_switch')
switch += svm
switch += tree

my_pipe += PipelineElement('StandardScaler')
my_pipe += switch

my_pipe.fit(X, y)

debug = True
