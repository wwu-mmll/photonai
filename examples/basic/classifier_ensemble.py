from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from photonai.base import Hyperpipe, PipelineElement, Stack
from photonai.optimization import FloatRange

X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(name='Estimator_pipe',
                    optimizer='random_grid_search',
                    metrics=['balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    inner_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    project_folder='./tmp/')

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# some feature selection
my_pipe += PipelineElement('LassoFeatureSelection',
                           hyperparameters={'percentile': FloatRange(start=0.1, step=0.1, stop=0.7, range_type='range'),
                                            'alpha': FloatRange(0.5, 1)},
                           test_disabled=True)

# add imbalanced group handling
my_pipe += PipelineElement('ImbalancedDataTransformer', method_name='SMOTE', test_disabled=False)

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

my_pipe.fit(X, y)
