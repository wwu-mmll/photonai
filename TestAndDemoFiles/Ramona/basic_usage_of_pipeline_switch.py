import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch
from sklearn.model_selection import KFold

""" MORE DOCUMENTATION CAN BE FOUND HERE:
 https://translap.atlassian.net/wiki/display/PENIS/Photon+Toolbox+Framework+HowTos
"""

# Load data
data_object = DataContainer()

# add values from ENIGMA thickness values to features
data_object += Features('../EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv',
                        usecols=np.arange(1, 73), na_values='NA')

# try to predict sex, which is column number 4
data_object += Targets('../EnigmaTestFiles/Covariates.csv', usecols=[4], na_values='NA')

# add age as covariate
data_object += Covariates('age', '../EnigmaTestFiles/Covariates.csv', usecols=[3], na_values='NA')

# example hyperparameter optimization for pipeline:
# 01. pca
# 02. support vector classifier OR logistic regression
cv_object = KFold(n_splits=3)

manager = Hyperpipe('god', cv_object, metrics=['accuracy',
                                               'confusion_matrix'], set_random_seed=True)

# add a pca analysis, specify hyperparameters to test
manager += PipelineElement.create('pca', {'n_components': [None, 10, 20], 'whiten': [True, False]},
                                  test_disabled=True)

# test to use a SVC
svc_estimator = PipelineElement.create('svc', {'C': np.arange(0.2, 1, 0.2), 'kernel': ['rbf', 'sigmoid']})
# or Logistic regression
lr_estimator = PipelineElement.create('logistic', {'C': np.logspace(-4, 4, 5)})

manager.add(PipelineSwitch('final_estimator', [svc_estimator, lr_estimator]))

# or whatever you want...
# the syntax is always: PipelineElement(Element identifier, hyperparameter dictionary, options to pass to the element)

# optimizes hyperparameters
X = data_object.features.values
y = np.ravel(data_object.targets.values)
manager.fit(X, y)

