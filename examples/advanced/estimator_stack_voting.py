from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

from photonai.base import Hyperpipe, Switch, Stack, Branch, PipelineElement
from photonai.optimization import IntegerRange

X, y = load_breast_cancer(return_X_y=True)

my_pipe = Hyperpipe('voting_example',
                    inner_cv=KFold(n_splits=10),
                    outer_cv=KFold(n_splits=5),
                    metrics=['balanced_accuracy', 'precision'],
                    best_config_metric='balanced_accuracy',
                    project_folder='./tmp')

my_pipe += PipelineElement("SimpleImputer")

switch = Switch("my_copy_switch")
switch += PipelineElement("StandardScaler")
switch += PipelineElement("RobustScaler", test_disabled=True)
my_pipe += switch

stack = Stack("EstimatorStack")

branch1 = Branch("SVMBranch")
branch1 += PipelineElement("PCA", hyperparameters={'n_components': IntegerRange(5, 10)})
branch1 += PipelineElement("SVC")

branch2 = Branch('TreeBranch')
branch2 += PipelineElement("PCA", hyperparameters={'n_components': IntegerRange(5, 10)})
branch2 += PipelineElement("DecisionTreeClassifier")

stack += branch1
stack += branch2
my_pipe += stack

my_pipe += PipelineElement("PhotonVotingClassifier")

my_pipe.fit(X, y)
