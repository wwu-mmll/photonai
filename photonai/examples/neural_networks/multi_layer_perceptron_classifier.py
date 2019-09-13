from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import IntegerRange
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
import time


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

settings = OutputSettings(project_folder='.')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='sk_opt',  # which optimizer PHOTON shall use
                    optimizer_params={'num_iterations': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=3),  # test each configuration ten times respectively,
                    verbosity=1,
                    output_settings=settings)


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

my_pipe += PipelineElement('PhotonMLPClassifier', hyperparameters={'layer_1': IntegerRange(0, 5),
                                                                   'layer_2': IntegerRange(0, 5),
                                                                   'layer_3': IntegerRange(0, 5)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True


