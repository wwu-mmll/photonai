import numpy as np
from loading import DataContainer, Features, Covariates, Targets
from HPOFramework.HPOBaseClasses import HyperpipeManager, PipelineElement

# Load data
data = DataContainer()
# load ENIGMA surface values
data += Features('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/CorticalMeasuresENIGMA_SurfAvg.csv',
                 usecols=np.arange(1, 73), na_values='NA')
# initial shape
print('feature shape before concat', data.features.data.shape)
# concatenate ENIGMA thickness values
data += Features('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/CorticalMeasuresENIGMA_ThickAvg.csv',
                 usecols=np.arange(1, 73), na_values='NA')
# shape after concat
print('feature shape after concat', data.features.data.shape)

# try to predict sex, which is column number 4
data += Targets('/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/Covariates.csv', usecols=[4], na_values='NA')

# data attribute = pandas data frame
print('data attribute returns:', type(data.targets.data))
# values attribute = data in form of ndarray
print('values attribute returns:', type(data.targets.values))

# add age as covariate
data += Covariates('age', '/home/rleenings/PycharmProjects/TFLearnTest/testDataFor/Covariates.csv',
                   usecols=[3], na_values='NA')
# items are accessible:
print(data.covariates['age'])

# example hyperparameter optimization for pipeline:
# 01. pca
# 02. keras neuronal net
keras_manager = HyperpipeManager(data)
# add a pca analysis, specify hyperparameters to test and default values
keras_manager += PipelineElement('pca', {'n_components': np.arange(10, 70, 10)})
# add a neural network with and try out x hidden layers with several sizes, set default values
keras_manager += PipelineElement('kdnn', {'hidden_layer_sizes': [[10], [5, 10], [10, 20, 10]]},
                           batch_normalization=True, learning_rate=0.3, target_dimension=10)
# optimize using grid_search
keras_manager.optimize('grid_search')
