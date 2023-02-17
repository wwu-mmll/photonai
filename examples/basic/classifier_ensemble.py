from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from photonai.base import Hyperpipe, PipelineElement, Stack

my_pipe = Hyperpipe(name='ensemble_pipe',
                    optimizer='random_grid_search',
                    metrics=['balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    inner_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')

# setup estimator stack
est_stack = Stack(name='classifier_stack')
clf_list = ['RandomForestClassifier', 'LinearSVC', 'NuSVC', "SVC", "MLPClassifier",
            "KNeighborsClassifier", "Lasso", "PassiveAggressiveClassifier", "LogisticRegression",
            "Perceptron", "RidgeClassifier", "SGDClassifier", "GaussianProcessClassifier",
            "AdaBoostClassifier", "BaggingClassifier", "GradientBoostingClassifier"]

for clf in clf_list:
    est_stack += PipelineElement(clf)
my_pipe += est_stack

my_pipe += PipelineElement('PhotonVotingClassifier')

X, y = load_breast_cancer(return_X_y=True)
my_pipe.fit(X, y)
