from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineStacking, PipelineBranch, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from photonai.configuration.Register import PhotonRegister
from photonai.investigator.Investigator import Investigator

from sklearn.model_selection import KFold

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)


mongo_settings = PersistOptions(mongodb_connect_url="mongodb://localhost:27017/photon_db",
                                save_predictions=False,
                                save_feature_importances=False,
                                json_filename="my_tree.json",
                                log_filename="my_tree.log")

PhotonRegister.info("SVC")


my_pipe = Hyperpipe('basic_svm_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    persist_options=mongo_settings)


svm = PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                   'C': FloatRange(0.5, 2, "linspace", num=3)})


tree = PipelineElement('DecisionTreeClassifier',  {'criterion': ['gini'],
                                                   'min_samples_split': IntegerRange(2, 4)})

branch = PipelineBranch('first_branch')
branch += PipelineElement('StandardScaler')
branch += PipelineElement('PCA', {'n_components': [5, 10]})
branch += tree

my_pipe_stack = PipelineStacking('final_stack', voting=True)
my_pipe_stack += svm
my_pipe_stack += branch

# my_pipe += PipelineStacking('final_stack', [svm, tree], voting=False)

my_pipe += my_pipe_stack

# my_pipe.fit(X, y)

# Investigator.show(my_pipe)
Investigator.load_from_file('/home/rleenings/Git/photon_core/photonai/examples/basic_svm_pipe_results.p')

debug = True
