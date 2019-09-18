from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from photonai.base import Hyperpipe, PipelineElement, Stack, OutputSettings
from photonai.optimization import IntegerRange

X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
settings = OutputSettings(project_folder='./tmp/')
my_pipe = Hyperpipe(name='Estimator_pipe',
                    optimizer='grid_search',
                    metrics=['balanced_accuracy'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    inner_cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                    output_settings=settings)

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# some feature selection
my_pipe += PipelineElement('CategorialANOVASelectPercentile',
                           hyperparameters={'percentile': IntegerRange(start=5, step=2, stop=20, range_type='range')},
                           test_disabled=True)

# add imbalanced group handling
my_pipe += PipelineElement('ImbalancedDataTransform', method_name='SMOTE', test_disabled=False)

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
# my_pipe += PipelineElement("RandomForestClassifier")
my_pipe.fit(X, y)
