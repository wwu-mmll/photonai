import numpy as np
from DataLoading.DataContainer import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import HyperpipeManager, PipelineElement, PipelineSwitch
from sklearn.model_selection import KFold

""" MORE DOCUMENTATION CAN BE FOUND HERE:
 https://translap.atlassian.net/wiki/display/PENIS/Photon+Toolbox+Framework+HowTos
"""

# Load data
data_object = DataContainer()
# load ENIGMA surface values
data_object += Features('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/CorticalMeasuresENIGMA_SurfAvg.csv',
                        usecols=np.arange(1, 73), na_values='NA')
# initial shape of loaded data:
print('feature shape before concat', data_object.features.data.shape)
# add values from another file, namely ENIGMA thickness values, to features
data_object += Features('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/CorticalMeasuresENIGMA_ThickAvg.csv',
                        usecols=np.arange(1, 73), na_values='NA')
# when adding more than one data source to features or targets, the data is internally concatenated horizontally
# --> see shape after concat
print('feature shape after concat', data_object.features.data.shape)

# try to predict sex, which is column number 4
data_object += Targets('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/Covariates.csv', usecols=[4],
                       na_values='NA')

# you can access the targets via data_objects.targets,
# and the features via data_objects.features,
# you can have the values as
# a) pandas data frame via the 'targets.data' attribute or
print('data attribute returns:', type(data_object.targets.data))
# b) numpy array via the 'targets.values' attribute
print('values attribute returns:', type(data_object.targets.values))

# add age as covariate
data_object += Covariates('age', '/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/Covariates.csv',
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
manager = HyperpipeManager(data_object, cv_object)

# add a pca analysis, specify hyperparameters to test
manager += PipelineElement.create('pca', {'n_components': np.arange(10, 70, 10), 'disable': [False, True]})

# add a neuronal net
# add a neural network, hyperparameters = try out x hidden layers with several sizes, set default values
# manager += PipelineElement('kdnn', {'hidden_layer_sizes': [[10], [5, 10], [10, 20, 10]]},
#                            batch_normalization=True, learning_rate=0.3, target_dimension=10)


# you can also use a SVC
svc_estimator = PipelineElement.create('svc', {'C': np.arange(0.2, 1, 0.2), 'kernel': ['linear', 'rbf']}, )
# or Logistic regression
lr_estimator = PipelineElement.create('logistic', {'C': np.logspace(-4, 4, 5)})

manager += PipelineSwitch('final_estimator', [svc_estimator, lr_estimator])

# or whatever you want...
# the syntax is always: PipelineElement(Element identifier, hyperparameter dictionary, options to pass to the element)

# then call optimize and specify optimization strategy
# optimize using grid_search
manager.optimize('grid_search')
