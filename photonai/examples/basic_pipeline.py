
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
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
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively,
                    verbosity=1,
                    cache_folder='/home/rleenings/Projects/TestNeuro/cache/',  # get error, warn and info message
                    # skips next folds of inner cv if accuracy and precision in first fold are below 0.96.
                    performance_constraints=[MinimumPerformance('accuracy', 0.96),
                                             DummyPerformance('precision', 0.2)])


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
# PhotonRegister.list()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
# PhotonRegister.info('SVC')

preprocessing = PreprocessingPipe()
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

my_pipe += PipelineElement('Graph', hyperparameters={'centrality__nr_nodes': IntegerRange(1, 3),
                                                     'centrality__huaihdsiud': Categorical(['linear', 'rbf']),
                                                     'xy__dhsud': FloatRange(0.5, 2)},
                           measures=["centrality", "xy", "a", "b", "c"])

# my_pipe += PipelineElement('LogisticRegression', hyperparameters={'penalty': ['l1', 'l2'], 'C': [0.5, 1]})

start_time = time.time()
# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
# AND SHOW THE RESULTS IN THE WEBBASED PHOTON INVESTIGATOR TOOL
# Investigator.show(my_pipe)
# Investigator.load_from_file(my_pipe.name, my_pipe.output_settings.local_file)


# YOU CAN ALSO SAVE THE BEST PERFORMING PIPELINE FOR FURTHER USE
# my_pipe.save_optimum_pipe('optimum_pipe.photon')


debug = True


