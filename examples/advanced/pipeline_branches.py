from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Stack, Branch
from photonai.optimization import IntegerRange, Categorical, FloatRange

X, y = load_breast_cancer(return_X_y=True)

my_pipe = Hyperpipe('basic_stacking',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='f1_score',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=10),
                    verbosity=1,
                    project_folder='./tmp/')

# BRANCH WITH QUANTILTRANSFORMER AND DECISIONTREECLASSIFIER
tree_qua_branch = Branch('tree_branch')
tree_qua_branch += PipelineElement('QuantileTransformer', n_quantiles=100)
tree_qua_branch += PipelineElement('DecisionTreeClassifier',
                                   {'min_samples_split': IntegerRange(2, 4)},
                                   criterion='gini')

# BRANCH WITH MinMaxScaler AND DecisionTreeClassifier
svm_mima_branch = Branch('svm_branch')
svm_mima_branch += PipelineElement('MinMaxScaler')
svm_mima_branch += PipelineElement('SVC',
                                   {'kernel': Categorical(['rbf', 'linear']),
                                    'C': FloatRange(0.01, 2.0, num=10)},
                                   gamma='auto')

# BRANCH WITH StandardScaler AND KNeighborsClassifier
knn_sta_branch = Branch('neighbour_branch')
knn_sta_branch += PipelineElement('StandardScaler')
knn_sta_branch += PipelineElement('KNeighborsClassifier')

# voting = True to mean the result of every branch
my_pipe += Stack('final_stack', [tree_qua_branch, svm_mima_branch, knn_sta_branch])

my_pipe += PipelineElement('LogisticRegression', solver='lbfgs')

my_pipe.fit(X, y)
