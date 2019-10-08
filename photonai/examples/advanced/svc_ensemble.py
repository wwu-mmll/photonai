from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Preprocessing, OutputSettings, Stack
from photonai.optimization import FloatRange, Categorical, IntegerRange

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='sk_opt',  # which optimizer PHOTON shall use
                    # optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=3),  # test each configuration ten times respectively,
                    output_settings=settings)


preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")

my_pipe += preprocessing

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)

my_stack = Stack("SVC_Ensemble")
for i in range(20):
    my_stack += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                        'C': FloatRange(0.5, 2, step=0.5)}, gamma='scale')
my_pipe += my_stack
my_pipe += PipelineElement('GaussianProcessClassifier')

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)



