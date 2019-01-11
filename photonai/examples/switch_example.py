from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from photonai.investigator.Investigator import Investigator
from sklearn.model_selection import KFold
from photonai.optimization.SpeedHacks import MinimumPerformance

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)

persist_options = OutputSettings(save_feature_importances='best',
                                 mongodb_connect_url="mongodb://trap-umbriel:27017/photon_db_test")

my_pipe = Hyperpipe('basic_switch_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1,
                    performance_constraints=MinimumPerformance('accuracy', 0.9),
                    output_settings=persist_options)


svm = PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

tree = PipelineElement('DecisionTreeClassifier')

switch = PipelineSwitch('estimator_switch')
switch += svm
switch += tree

my_pipe += PipelineElement('StandardScaler')
my_pipe += switch

my_pipe.fit(X, y)


Investigator.show(my_pipe)

debug = True
