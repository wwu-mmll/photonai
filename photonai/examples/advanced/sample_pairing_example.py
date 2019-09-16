from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('sample_pairing_example_classification',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=5),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively
                    verbosity=1) # get error, warn and info message                    )


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# add sample pairing
my_pipe += PipelineElement('SamplePairingClassification', {'draw_limit': [500, 1000, 10000],
                                                       'generator': Categorical(['nearest_pair', 'random_pair'])},
                           distance_metric='euclidean', test_disabled=True)

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('RandomForestClassifier', hyperparameters={'n_estimators': [10]})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True


