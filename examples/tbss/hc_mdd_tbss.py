from sklearn.model_selection import StratifiedKFold
from photonai.optimization import IntegerRange
from photonai.base import Hyperpipe, PipelineElement, Switch, Preprocessing, CallbackElement
from photonai_neuro import NeuroBranch
import pandas as pd
from os.path import join

# (0 = MDD, 1 = BD)

# def inspect_shape(X, y=None, **kwargs):
#     print(X.shape)

praefix = '/spm-data/vault-data4/FOR2107/SPM/_2ndLevel_Kathi/BDvsMDD_150121/analyses/ML_Analyse/'
data_csv = pd.read_csv(join(praefix,'Gruppenzugehoerigkeit.csv'), delimiter=";")
proband = data_csv["Proband"].to_numpy()
targets = data_csv["Gruppe"].to_numpy()
features = [join(praefix, "TBSS_FA_{}.nii.gz".format(str(p).zfill(4))) for p in proband]

# CREATE HYPERPIPE
my_pipe = Hyperpipe('basic_switch_pipe',
                    optimizer='random_grid_search',
                    optimizer_params={'n_configurations': 30},
                    metrics=['balanced_accuracy', 'precision', 'recall'],
                    best_config_metric='balanced_accuracy',
                    outer_cv=StratifiedKFold(n_splits=10, shuffle=True),
                    inner_cv=StratifiedKFold(n_splits=10, shuffle=True),
                    verbosity=1,
                    project_folder='./tmp/')


pre = Preprocessing()
neuro_branch = NeuroBranch('NeuroBranch')
custom_mask = join(praefix, 'mean_FA_skeleton_mask.nii.gz')
neuro_branch += PipelineElement('BrainMask', mask_image=custom_mask, extract_mode='vec', batch_size=20)
pre += neuro_branch
my_pipe += pre

# Transformer Switch
my_pipe += PipelineElement('StandardScaler')
# my_pipe += CallbackElement("InspectShape", delegate_function=inspect_shape)

my_pipe += PipelineElement('FRegressionSelectPercentile', hyperparameters={'percentile': [5, 10, 30, 50]},
                           test_disabled=True)
my_pipe += PipelineElement('PCA', test_disabled=True)
# Estimator Switch
svm = PipelineElement('SVC',
                      hyperparameters={'kernel': ['rbf', 'linear'],
                                       'C': [0.1, 1, 2, 5, 10, 25, 50, 100]})

tree = PipelineElement('DecisionTreeClassifier',
                       hyperparameters={'min_samples_split': IntegerRange(2, 5),
                                        'min_samples_leaf': IntegerRange(1, 5),
                                        'criterion': ['gini', 'entropy']})

my_pipe += Switch('EstimatorSwitch', [svm, tree])

my_pipe.fit(features, targets)