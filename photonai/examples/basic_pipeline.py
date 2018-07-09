
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.optimization.SpeedHacks import MinimumPerformance
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
# mongo_settings = PersistOptions(mongodb_connect_url="mongodb://localhost:27017/photon_db",
#                                 save_predictions=False,
#                                 save_feature_importances=False)


# save_options = PersistOptions(local_file="/home/photon_user/photon_test/test_item.p")


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
                    verbosity=1) # get error, warn and info message                    )


# SHOW WHAT IS POSSIBLE IN THE CONSOLE
PhotonRegister.list()

# NOW FIND OUT MORE ABOUT A SPECIFIC ELEMENT
PhotonRegister.info('SVC')


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': [5, 10, None]}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.5, 2, "linspace", num=5)})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# AND SHOW THE RESULTS IN THE WEBBASED PHOTON INVESTIGATOR TOOL
#Investigator.show(my_pipe)

# YOU CAN ALSO SAVE THE BEST PERFORMING PIPELINE FOR FURTHER USE
my_pipe.save_optimum_pipe('optimum_pipe.photon')

# YOU CAN ALSO LOAD YOUR RESULTS FROM THE MONGO DB
# Investigator.load_from_db(mongo_settings.mongodb_connect_url, my_pipe.name)

debug = True


