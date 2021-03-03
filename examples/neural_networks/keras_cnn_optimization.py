# optimized cnn model with PHOTONAI
# content by J. Brownlee:
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# HAR-Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
# required file: dataset.py from examples/neural_network

import os

from keras.utils import data_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from examples.neural_networks.dataset import load_har

from photonai.base import Hyperpipe, PipelineElement, PhotonRegistry
from photonai.modelwrapper.keras_base_models import KerasBaseClassifier
from photonai.optimization import IntegerRange, BooleanSwitch

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


# Transformer and Estimator Definition
class MyCnnScaler(BaseEstimator):

    def __init__(self, standardize: bool = True,):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.standardize = standardize

    def fit(self, data, targets=None, **kwargs):
        """
        Adjust the underlying model or method to the data.

        Returns
        -------
        IMPORTANT: must return self!
        """
        # remove overlap
        cut = int(X.shape[1] / 2)
        longX = X[:, -cut:, :]
        # flatten windows
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        # flatten train and test
        if self.standardize:
            self.s = StandardScaler()
            # fit on training data
            self.s.fit(longX)
        return self

    def transform(self, X, targets=None, **kwargs):
        """
        Apply the method's logic to the data.
        """
        # flatten train and test
        flatX = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        # standardize
        if self.standardize:
            # apply to training and test data
            flatX = self.s.transform(flatX)
        # reshape
        flatX = flatX.reshape((X.shape))
        return flatX


class MyOptimizedCnnEstimator(KerasBaseClassifier):

    def __init__(self, n_filters: int = 64,
                 kernel_size: int = 3,
                 epochs: int = 10,
                 verbosity: int = 1):
        # it is important that you name your params the same in the constructor
        #  stub as well as in your class variables!
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.verbosity = verbosity

    def fit(self, X, y):
        # the optimized parameters are not given directly to the constructor.
        # Therefore the model can only be created in fit.
        model = self.build_model(self.n_filters, self.kernel_size, X.shape[1], X.shape[2], 6)
        super(MyOptimizedCnnEstimator, self).__init__(model=model,
                                                      epochs=self.epochs,
                                                      nn_batch_size=32,
                                                      verbosity=self.verbosity,
                                                      multi_class=True)
        super(MyOptimizedCnnEstimator, self).fit(X, y)

    @staticmethod
    def build_model(n_filters, kernel_size, n_timesteps, n_features, n_outputs):
        model = Sequential()
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,
                                                                                                     n_features)))
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
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
registry.register(photon_name='MyCnnScaler',
                  class_str='keras_cnn_optimization.MyCnnScaler',
                  element_type='Transformer')

registry.register(photon_name='MyOptimizedCnnEstimator',
                  class_str='keras_cnn_optimization.MyOptimizedCnnEstimator',
                  element_type='Estimator')
# This needs to be done every time you run the script
registry.activate()

if __name__ == "__main__":  # prevents double optimization, cause registry calls this file again

    # DESIGN YOUR PIPELINE
    my_pipe = Hyperpipe('cnn_keras_multiclass_pipe',
                        optimizer='sk_opt',
                        optimizer_params={'n_configurations': 25},
                        metrics=['accuracy'],
                        best_config_metric='accuracy',
                        outer_cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
                        inner_cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
                        verbosity=1,
                        project_folder='./tmp/')

    my_pipe += PipelineElement('MyCnnScaler', hyperparameters={'standardize': BooleanSwitch()})

    my_pipe += PipelineElement('MyOptimizedCnnEstimator',
                               hyperparameters={'n_filters': IntegerRange(8, 256),
                                                'kernel_size': IntegerRange(2, 11)},
                               epochs=10, verbosity=0)

    # NOW TRAIN YOUR PIPELINE
    my_pipe.fit(X, y)
