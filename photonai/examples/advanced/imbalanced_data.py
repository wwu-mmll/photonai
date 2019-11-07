from sklearn.model_selection import KFold
from imblearn.datasets import fetch_datasets

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import FloatRange, Categorical, IntegerRange

# example of imbalanced dataset
dataset = fetch_datasets()['coil_2000']
X, y = dataset.data, dataset.target
# ratio class 0: 0.06%, class 1: 0.94%

my_pipe = Hyperpipe('basic_svm_pipe_no_performance',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 5},
                    metrics=['accuracy', 'precision', 'recall'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=5),
                    verbosity=1,
                    output_settings=OutputSettings(project_folder='./tmp/'))


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': IntegerRange(5, 20)},
                           test_disabled=True)

my_pipe += PipelineElement('ImbalancedDataTransform',
                           hyperparameters={'method_name': Categorical(['RandomUnderSampler',
                                                                        'SMOTE'])})

my_pipe += PipelineElement('SVC',
                           hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                            'C': FloatRange(0.5, 2)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# imbalance_type = OVERSAMPLING:
#     - RandomOverSampler
#     - SMOTE
#     - ADASYN
#
# imbalance_type = UNDERSAMPLING:
#     - ClusterCentroids,
#     - RandomUnderSampler,
#     - NearMiss,
#     - InstanceHardnessThreshold,
#     - CondensedNearestNeighbour,
#     - EditedNearestNeighbours,
#     - RepeatedEditedNearestNeighbours,
#     - AllKNN,
#     - NeighbourhoodCleaningRule,
#     - OneSidedSelection
#
# imbalance_type = COMBINE:
#     - SMOTEENN,
#     - SMOTETomek
