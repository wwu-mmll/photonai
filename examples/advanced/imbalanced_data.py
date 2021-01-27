from sklearn.model_selection import KFold
from imblearn.datasets import fetch_datasets

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical, IntegerRange

# example of imbalanced dataset
dataset = fetch_datasets()['coil_2000']
X, y = dataset.data, dataset.target
# ratio class 0: 0.06%, class 1: 0.94%

my_pipe = Hyperpipe('basic_svm_pipe_no_performance',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='recall',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': IntegerRange(5, 20)},
                           test_disabled=True)

my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name': Categorical(['RandomUnderSampler','SMOTE'])},
                           test_disabled=True)

my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(0.5, 2)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# Possible values for method_name:
# imbalance_type = OVERSAMPLING:
#        - ADASYN
#        - BorderlineSMOTE
#        - KMeansSMOTE
#        - RandomOverSampler
#        - SMOTE
#        - SMOTENC
#        - SVMSMOTE
#
#    imbalance_type = UNDERSAMPLING:
#        - ClusterCentroids,
#        - RandomUnderSampler,
#        - NearMiss,
#        - InstanceHardnessThreshold,
#        - CondensedNearestNeighbour,
#        - EditedNearestNeighbours,
#        - RepeatedEditedNearestNeighbours,
#        - AllKNN,
#        - NeighbourhoodCleaningRule,
#        - OneSidedSelection
#
#    imbalance_type = COMBINE:
#        - SMOTEENN,
#        - SMOTETomek
