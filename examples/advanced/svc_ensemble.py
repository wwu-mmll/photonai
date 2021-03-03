from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing, Stack
from photonai.optimization import FloatRange, Categorical

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('svm_ensemble_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    project_folder='./tmp/')


preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing

my_pipe.add(PipelineElement('StandardScaler'))

my_stack = Stack("SVC_Ensemble")
for i in range(20):
    my_stack += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                        'C': FloatRange(0.5, 2, step=0.5)}, gamma='scale')
my_pipe += my_stack
my_pipe += PipelineElement('GaussianProcessClassifier')

my_pipe.fit(X, y)
