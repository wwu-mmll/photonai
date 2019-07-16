
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.optimization.SpeedHacks import MinimumPerformance
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# targets have to be 1 and -1
y[y == 0] = -1

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe_no_performance',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy', 'precision', 'recall'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively
                    # skips next folds of inner cv if accuracy and precision in first fold are below 0.96.
                    performance_constraints=[MinimumPerformance('accuracy', 0.96),
                                             MinimumPerformance('precision', 0.96)],
                    verbosity=1)  # get error, warn and info message                    )

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('AnomalyDetectorWrapper', hyperparameters={'wrapped_estimator': ['OneClassSVM'],
                                                                      'kernel': ['linear', 'rbf']})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True
