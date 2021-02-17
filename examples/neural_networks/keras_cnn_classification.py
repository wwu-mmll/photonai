# cnn model with PHOTONAI
# content by J. Brownlee:
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# HAR-Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# required file: data.py from examples/neural_network

from keras.utils import data_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit

from examples.neural_networks.dataset import load_har

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical

dataset_path = data_utils.get_file(
    fname='UCI HAR Dataset.zip',
    origin='https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
    file_hash='53e099237392e0b9602f8c38f578bd8f',
    hash_algorithm='md5',
    cache_subdir='photonai_datasets',
    extract=True,
    archive_format='zip'
)

X, y = load_har(prefix=dataset_path.replace('.zip', ''))

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

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('cnn_keras_multiclass_pipe',
                    optimizer='grid_search',
                    optimizer_params={},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2),
                    inner_cv=StratifiedShuffleSplit(n_splits=2, test_size=0.2),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe += PipelineElement('KerasBaseClassifier',
                           hyperparameters={'epochs': Categorical([10, 20])},
                           verbosity=1,
                           model=model)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
