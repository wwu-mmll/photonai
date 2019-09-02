
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, Preprocessing
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.optimization.SpeedHacks import MinimumPerformance, DummyPerformance
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
import time

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='random_grid_search',  # which optimizer PHOTON shall use
                    optimizer_params={'k': 10},
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=3),  # test each configuration ten times respectively,
                    verbosity=1,
                    cache_folder='/home/rleenings/Projects/TestNeuro/cache/')  # get error, warn and info message
                    # skips next folds of inner cv if accuracy and precision in first fold are below 0.96.
                    # performance_constraints=[MinimumPerformance('accuracy', 0.96),
                    #                          DummyPerformance('precision', 0.2)])


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
# PhotonRegister.list()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
# PhotonRegister.info('SVC')

preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")

my_pipe += preprocessing

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2)}, gamma='scale')
# my_pipe += PipelineElement('LogisticRegression', hyperparameters={'penalty': ['l1', 'l2'], 'C': [0.5, 1]})


# NOW TRAIN YOUR PIPELINE
start_time = time.time()
my_pipe.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

Investigator.show(my_pipe)
debug = True


