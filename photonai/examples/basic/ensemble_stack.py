from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Stack
from photonai.optimization import IntegerRange, DummyPerformance
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(name='Estimator_pipe',  # the name of your pipeline
                    optimizer='grid_search',
                    metrics=['balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='balanced_accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # repeat hyperparameter search three times
                    inner_cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))  # test each configuration ten times respectively,
                    # skips next folds of inner cv if balanced_accuracy is not at least .2 better than dummy
                    #performance_constraints=[DummyPerformance('balanced_accuracy', .2)])


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
