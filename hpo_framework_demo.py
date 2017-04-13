import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch
from sklearn.model_selection import KFold

""" MORE DOCUMENTATION CAN BE FOUND HERE:
 https://translap.atlassian.net/wiki/display/PENIS/Photon+Toolbox+Framework+HowTos
"""

# Load data
data_object = DataContainer()

# add values from ENIGMA thickness values to features
data_object += Features('EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv',
                        usecols=np.arange(1, 73), na_values='NA')

# try to predict sex, which is column number 4
data_object += Targets('EnigmaTestFiles//Covariates.csv', usecols=[4],
                       na_values='NA')

# you can access the targets via data_objects.targets,
# and the features via data_objects.features,
# you can have the values as
# a) pandas data frame via the 'targets.data' attribute or
print('data attribute returns:', type(data_object.targets.data))
# b) numpy array via the 'targets.values' attribute
print('values attribute returns:', type(data_object.targets.values))

# add age as covariate
data_object += Covariates('age', 'EnigmaTestFiles//Covariates.csv',
                          usecols=[3], na_values='NA')

# covariate items are accessible via data_objects.covariates by their name:
print(data_object.covariates['age'])
# again you can have
# a) a pandas dataframe: data_object.covariates['age'].data or
# b) a numpy array: data_object.covariates['age'].values

# example hyperparameter optimization for pipeline:
# 01. pca
# 02. keras neuronal net  OR support vector classifier OR logistic regression
cv_object = KFold(n_splits=3)

manager = Hyperpipe('god', cv_object)

# add a pca analysis, specify hyperparameters to test
manager += PipelineElement.create('pca', {'n_components': [None, 10, 20], 'whiten': [True, False]},
                                  set_disabled=True)

# ADD ANY KERAS MODEL LIKE SHOWN IN WRAPPERMODEL CLASS
# manager += PipelineElement.create('wrapper_model', {'learning_rate': [0.1, 0.2, 0.3]})

# add a neuronal net
# add a neural network, hyperparameters = try out x hidden layers with several sizes, set default values
# kdnn = PipelineElement.create('kdnn', {'hidden_layer_sizes': [[10, 5], [5]]},
#                               batch_normalization=True, learning_rate=0.3, target_dimension=10)

# you can also use a SVC
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

final_predictor = manager.optimum_pipe
final_predictor.fit(X, y)
prediction = final_predictor.predict(X)

# access the performance and config histories
config1 = manager.config_history[0]
performance1 = manager.performance_history[0]

