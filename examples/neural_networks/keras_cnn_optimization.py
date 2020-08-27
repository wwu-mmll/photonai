# optimized cnn model with PHOTONAI
# example: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# HARDataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# required file: data.py from examples/neural_network
import os

from keras.utils import data_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import KFold

from examples.neural_networks.data import load_dataset

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, PhotonRegistry
from photonai.modelwrapper.keras_base_models import KerasBaseClassifier
from photonai.optimization import IntegerRange

dataset_path = data_utils.get_file(
    fname='UCI HAR Dataset.zip',
    origin='https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
    file_hash='53e099237392e0b9602f8c38f578bd8f',
    hash_algorithm='md5',
    cache_subdir='photonai_datasets',
    extract=True,
    archive_format='zip'
)

X, y = load_dataset(prefix=dataset_path.replace('.zip', ''))


class MyOptimizedCnnEstimator(KerasBaseClassifier):

    def __init__(self, n_filters: int = 64, epochs: int = 10, verbosity: int = 1):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        model = self.build_model(n_filters, X.shape[1], X.shape[2], 6)
        super(MyOptimizedCnnEstimator, self).__init__(model=model,
                                                      epochs=epochs,
                                                      nn_batch_size=32,
                                                      multi_class=True,
                                                      verbosity=verbosity)

    @classmethod
    def build_model(cls, n_filters, n_timesteps, n_features, n_outputs):
        model = Sequential()
        model.add(Conv1D(filters=n_filters, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=n_filters, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


# REGISTER ELEMENT
base_folder = os.path.dirname(os.path.abspath(__file__))
custom_elements_folder = os.path.join(base_folder, '')

registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)

# This needs to be done only once on your device
registry.register(photon_name='MyOptimizedCnnEstimator',
                  class_str='keras_cnn_optimization.MyOptimizedCnnEstimator',
                  element_type='Estimator')

# This needs to be done every time you run the script
registry.activate()

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('cnn_keras_multiclass_pipe',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 10},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=2),
                    verbosity=1,
                    output_settings=OutputSettings(project_folder='./tmp/'))

my_pipe += PipelineElement('MyOptimizedCnnEstimator',
                           hyperparameters={'n_filters': IntegerRange(8, 256)},
                           epochs=3, verbosity=1)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
