import os
from six.moves import cPickle as pickle

from HPOFramework.HPOBaseClasses import HyperpipeManager


data_root = '/home/rleenings/PycharmProjects/TFLearnTest/'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
all_data = pickle.load(open(pickle_file, "rb"))

train_data = all_data['train_dataset']
train_labels = all_data['train_labels']
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

X_digits = train_data[1:2000, :]
y_digits = train_labels[1:2000]

# parameters to optimize
n_components = [60, 80, 100, 120]
grad_desc_alphas = [0.1, 0.3, 0.5]
param_list = dict(pca__n_components=n_components, dnn__gd_alpha=grad_desc_alphas)

manager = HyperpipeManager(param_list, X_digits, y_digits)
manager.optimize('grid_search')

