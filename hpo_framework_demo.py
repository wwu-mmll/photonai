import os
from six.moves import cPickle as pickle

from HPOFramework.HPOBaseClasses import HyperpipeManager, PipelineElement


# load data
data_root = '/home/rleenings/PycharmProjects/TFLearnTest/'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
all_data = pickle.load(open(pickle_file, "rb"))

train_data = all_data['train_dataset']
train_labels = all_data['train_labels']
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

X_digits = train_data[0:2000, :]
y_digits = train_labels[0:2000]


# Example A
manager = HyperpipeManager(X_digits, y_digits)
manager += PipelineElement('pca', {'n_components': [20, 60, 80]})
manager += PipelineElement('dnn', {'gd_alpha': [0.1, 0.3, 0.5]}, gd_alpha=0.1)
scores = manager.optimize('grid_search')

# Example B
# parameters to optimize
pipeline_config = {'pca': {'n_components': [20, 60, 80]},
                   'dnn': {'gd_alpha': [0.1, 0.3, 0.5]}}
config_manager = HyperpipeManager(X_digits, y_digits, config=pipeline_config)
scores2 = config_manager.optimize('grid_search')

