from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineStacking, PipelineBranch, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)


mongo_settings = PersistOptions(mongodb_connect_url="mongodb://localhost:27017/photon_db",
                                save_predictions='best',
                                save_feature_importances='best',
                                local_file="my_tree.p",
                                log_filename="my_tree.log")


my_pipe = Hyperpipe('basic_stacking',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1)

tree_branch = PipelineBranch('first_branch')
tree_branch += PipelineElement('StandardScaler')
tree_branch += PipelineElement('DecisionTreeClassifier', {'criterion': ['gini'],
                                                          'min_samples_split': IntegerRange(2, 4)})

svm_branch = PipelineBranch('svm_branch')
svm_branch += PipelineElement('StandardScaler')
svm_branch += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                      'C': FloatRange(0.5, 2, "linspace", num=3)})


my_pipe_stack = PipelineStacking('final_stack', voting=False)
my_pipe_stack += svm_branch
my_pipe_stack += tree_branch

my_pipe += my_pipe_stack

my_pipe += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear'])})

my_pipe.fit(X, y)

# Investigator.load_from_file("basic_svm_pipe", 'my_tree.p')

debug = True
