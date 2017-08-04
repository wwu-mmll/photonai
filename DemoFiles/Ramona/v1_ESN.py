import os
import pickle
import numpy as np
import pandas as pd
from Framework.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import preprocessing
from DataLoading.DataLoader import MatLoader
import matplotlib.pyplot as plt

# # LOAD DATA
# folder = '/media/truecrypt1/Matlab/daten/Alle/'
# pickle_file_name = "data_per_day.pickle"
#
# patient_data = np.zeros((0, 24*60*60))
# future_patient_data = np.zeros((0, 24*60*60))
# patient_labels = np.zeros((0, 1))
#
# max_num_patients_to_calculate = 2
# patient_cnt = -1

# read_patient_data = False
#
# if read_patient_data:
#     for file in os.listdir(folder):
#          if file.endswith(".txt"):
#              patient_cnt += 1
#              if patient_cnt < max_num_patients_to_calculate:
#
#                 print('reading file:' + str(patient_cnt), file)
#                 data_frame = pd.read_csv(folder+file, sep=';', header=None, skiprows=5)
#                 # print('...converting to datetime') # pd.to_datetime
#                 time = data_frame.iloc[:, 0].values
#                 time = np.array([time[i[0]][0:8] for i in enumerate(time)])
#                 test_data = data_frame.iloc[:, 1].values
#
#                 # find first and last midnight, take data in between, trash others
#                 midnight_index = np.where(time == '00:00:00')
#                 test_data = test_data[midnight_index[0][0]:midnight_index[0][-1]]
#                 future_data = np.insert(test_data, 0, 0)
#                 future_data = np.delete(test_data, -1, 0)
#                 data_per_day = np.reshape(test_data, (13, 24*60*60))
#                 future_data_per_day = np.reshape(future_data, (13, 24*60*60))
#                 patient_data = np.vstack((patient_data, data_per_day))
#                 future_patient_data = np.vstack((future_patient_data, future_data_per_day))
#                 patient_labels = np.vstack((patient_labels, np.ones((data_per_day.shape[0], 1))*patient_cnt))
#
#     # save data
#     pickle_file = open(pickle_file_name, "wb")
#     patient_data = preprocessing.normalize(preprocessing.scale(patient_data), norm='l1')
#     pickle.dump((patient_data, patient_labels, future_patient_data), pickle_file)
#     pickle_file.close()
#
#     X = patient_data
#     y = patient_labels
#     f = future_patient_data
# else:
#     pickle_file = open(pickle_file_name, "rb")
#     loaded_data = pickle.load(pickle_file)
#     y = loaded_data[1]
#     X = loaded_data[0]
#     f = loaded_data[2]
#
#     # subsampling
#     cols = X.shape[1]
#     X = X[:, 0:cols:5*60]
#     f = f[:, 0:cols:5*60]


# dc = MatLoader()
# mat_vals = dc('/media/truecrypt1/Matlab/daten/PSGMat/EDSConHyper/Freese, Daniel.mat')
#
#
# def plot_some_data(data, targets):
#     ax_array = np.arange(0, data.shape[0], 1)
#     plt.figure().clear()
#     plt.plot(ax_array, data, ax_array, targets)
#     plt.title('A sample of data')
#     plt.show()
#
# labels = mat_vals['labels']
# labels = np.array(labels).reshape((-1, 1)).flatten()
#
# data = mat_vals['data']
# data = np.array(data).reshape((-1, 1)).flatten()
# data[data > 1500] = 1500
# # data[data == 0] = 0.01
# data = preprocessing.maxabs_scale(data)
# plot_some_data(data[0:24*60*60], labels[0:24*60*60])
#
# epoch_length = 30
# epoch_data = data.reshape((-1, 30))
# epoch_labels = mat_vals['epochLabels']
# # epoch_labels = np.array(epoch_labels) + 1
#
# X = epoch_data
# y = epoch_labels


X = np.array(np.loadtxt('/home/rleenings/Projects/ESN/MackeyGlass_t17.txt')).reshape((-1, 1))
y = X

# BUILD PIPELINE
manager = Hyperpipe('test_manager',
                    hyperparameter_search_cv_object=ShuffleSplit(test_size=0.2, n_splits=1),
                    hyperparameter_specific_config_cv_object=KFold(n_splits=3, shuffle=False),
                    metrics=['mean_squared_error'], logging=False)

manager.add(PipelineElement.create('py_esn_r', hyperparameters={'reservoir_size': [100, 500, 1000],
                                                                'leaking_rate': [0.3, 0.5, 0.7],
                                                                'spectral_radius': [1.25, 0.75, 0.9, 0.5]},
                                   init_len=0))
manager.fit(X, y)


# THE END
debugging = True
