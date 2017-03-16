import os
import numpy as np
from six.moves import cPickle as pickle

from HPOFramework.HPOBaseClasses import HyperpipeManager, PipelineElement


# load data
data_root = '/home/rleenings/PycharmProjects/TFLearnTest/'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
all_data = pickle.load(open(pickle_file, "rb"))

train_data = all_data['train_dataset']
train_labels = all_data['train_labels']
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

# randomize
permutation = np.random.permutation(train_labels.shape[0])
X_items = train_data[permutation, :]
y_items = train_labels[permutation]

# take and only x items
X_digits = X_items[0:2000, :]
y_digits = y_items[0:2000]

# Example A: code syntax
manager = HyperpipeManager(X_digits, y_digits)
# add a PCA and try out several numbers of components
manager += PipelineElement('pca', {'n_components': [20, 60, 80]})
manager += PipelineElement('dnn', {'gd_alpha': [0.1, 0.3, 0.5]}, gd_alpha=0.1)
scores = manager.optimize('grid_search')

# use best params to score new data
# new_X_digits = X_items[5000:6000]
# score_of_new_data = manager.optimum_pipe.fit_predict(new_X_digits)

# Example B: config syntax
# parameters to optimize
pipeline_config = {'pca': {'n_components': [20, 60, 80]},
                   'dnn': {'gd_alpha': [0.1, 0.3, 0.5]}}
config_manager = HyperpipeManager(X_digits, y_digits, config=pipeline_config)
# scores2 = config_manager.optimize('grid_search')

# Example C: keras neuronal net
keras_manager = HyperpipeManager(X_digits, y_digits)
keras_manager += PipelineElement('pca', {'n_components': [20, 60, 80]})
# add a neural network with and try out x hidden layers with several sizes
keras_manager += PipelineElement('kdnn', {'hidden_layer_sizes': [[10], [5, 10], [10, 20, 10]]},
                           batch_normalization=True, learning_rate=0.3, target_dimension=10)
