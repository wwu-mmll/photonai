# cnn model with PHOTON
# example: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# HARDataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

from numpy import dstack
from pandas import read_csv
import numpy as np
from keras.utils import data_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import KFold
from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import Categorical


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + '/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + '/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return np.concatenate((trainX, testX), axis=0), np.concatenate((trainy, testy), axis=0)

dataset_path = data_utils.get_file(
    fname='UCI HAR Dataset.zip',
    origin='https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
    file_hash='53e099237392e0b9602f8c38f578bd8f',
    hash_algorithm='md5',
    cache_subdir='photonai_datasets',
    extract=True,
    archive_format='zip'
)

X, y = load_dataset(prefix=dataset_path.replace('.zip', ''))  # your path to your "HARDataset" folder (download required, link above)

verbose, epochs, batch_size = 1, 10, 32
n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], 6
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

settings = OutputSettings(project_folder='./tmp/')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('cnn_keras_multiclass_pipe',
                    optimizer='grid_search',
                    optimizer_params={},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=2),
                    verbosity=1,
                    output_settings=settings)

my_pipe += PipelineElement('KerasBaseClassifier',
                           hyperparameters={'epochs': Categorical([10, 20])},
                           verbosity=1,
                           model=model)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
