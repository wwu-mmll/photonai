import warnings
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.datasets import fetch_datasets

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical

# Since we test very imbalanced data, we want to ignore some metric based zero-divisions.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# example of imbalanced dataset
dataset = fetch_datasets()['coil_2000']
X, y = dataset.data, dataset.target
# ratio class 0: 6%, class 1: 94%

my_pipe = Hyperpipe('balancing_pipe',
                    optimizer='grid_search',
                    metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                    best_config_metric='f1_score',
                    outer_cv=StratifiedKFold(n_splits=3),
                    inner_cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
                    verbosity=1,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('StandardScaler')

tested_methods = Categorical(['RandomOverSampler', 'SMOTEENN', 'SVMSMOTE',
                              'BorderlineSMOTE', 'SMOTE', 'ClusterCentroids'])
my_pipe += PipelineElement('ImbalancedDataTransformer',
                           hyperparameters={'method_name': tested_methods},
                           test_disabled=True)

my_pipe += PipelineElement("RandomForestClassifier", n_estimators=200)

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
# imbalance_type = UNDERSAMPLING:
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
# imbalance_type = COMBINE:
#        - SMOTEENN,
#        - SMOTETomek
