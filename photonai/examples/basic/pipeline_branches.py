from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineStacking, PipelineBranch
from photonai.optimization.Hyperparameters import FloatRange, IntegerRange, Categorical
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(True)


my_pipe = Hyperpipe('basic_stacking',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='f1_score',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1)

# BRANCH WITH QUANTILTRANSFORMER AND DECISIONTREECLASSIFIER
tree_qua_branch = PipelineBranch('tree_branch')
tree_qua_branch += PipelineElement('QuantileTransformer')
tree_qua_branch += PipelineElement('DecisionTreeClassifier',{'min_samples_split': IntegerRange(2, 4)},criterion='gini')

# BRANCH WITH MinMaxScaler AND DecisionTreeClassifier
svm_mima_branch = PipelineBranch('svm_branch')
svm_mima_branch += PipelineElement('MinMaxScaler')
svm_mima_branch += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']),
                                      'C':2.0},gamma='auto')

# BRANCH WITH StandardScaler AND KNeighborsClassifier
knn_sta_branch = PipelineBranch('neighbour_branch')
knn_sta_branch += PipelineElement('StandardScaler')
knn_sta_branch += PipelineElement('KNeighborsClassifier')

# voting = True to mean the result of every branch
my_pipe += PipelineStacking('final_stack', [tree_qua_branch, svm_mima_branch, knn_sta_branch], voting=True)

my_pipe += PipelineElement('LogisticRegression', solver='lbfgs')

my_pipe.fit(X, y)
