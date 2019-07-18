"""
Basic example for accessing the PHOTON results tree
"""
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PreprocessingPipe
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
from photonai.validation.ResultsDatabase import MDBHelper
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
import numpy as np


# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_breast_cancer(True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=2),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=3),  # test each configuration ten times respectively,
                    verbosity=1)  # get error, warn and info message


preprocessing = PreprocessingPipe()
preprocessing += PipelineElement("LabelEncoder")

my_pipe += preprocessing

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 10)}, test_disabled=True)
# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf', 'linear']),
                                                   'C': FloatRange(0.001, 1, range_type='logspace', num=5)}, gamma='scale')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# ACCESS THE RESULTS TREE
results = my_pipe.result_tree
# handler = ResultsTreeHandler(results)
#
# # get performance of outer folds
# performance_outer_folds = handler.get_performance_outer_folds()

# get performance of every configuration


tested_configs_outer_folds = list()
for outer_fold in results.outer_folds:
    accs = list()
    bal_accs = list()
    for config in outer_fold.tested_config_list:
        accs.append(MDBHelper.get_metric(config, 'FoldOperations.MEAN', 'accuracy', train=False))
        bal_accs.append(MDBHelper.get_metric(config, 'FoldOperations.MEAN', 'balanced_accuracy', train=False))
    tested_configs_outer_folds.append({'accs': accs, 'bal_accs': bal_accs})
print(tested_configs_outer_folds)

import matplotlib.pylab as plt
plt.figure()
for outer_fold in tested_configs_outer_folds:
    plt.plot(np.arange(0, len(outer_fold['accs'])), outer_fold['accs'])
plt.show()

plt.figure()
for outer_fold in tested_configs_outer_folds:
    min_config = list()
    last_config = 1
    for config in outer_fold['accs']:
        if config < last_config:
            last_config = config
        min_config.append(last_config)
    plt.plot(np.arange(0, len(outer_fold['accs'])), min_config)
plt.show()

debug = True


