import keras
import keras.optimizers
import numpy as np
import tensorflow as tf
from keras.layers import Dropout, Dense, LSTM
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit

from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint



class CNN1d_Autoencoder(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self.time_window_size = None
        self.metric = None
        self.threshold = 5.0
        self.config = None
        self.VERBOSE = 1

        self.batch_size = 8
        self.epochs = 100
        self.validation_split = 0.1
        self.metric = 'mean_absolute_error'
        self.estimated_negative_sample_ratio = 0.9

        self.x = None
        self.y_ = None
        self.model = None

    def predict(self, X):
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        if self.target_dimension > 1:
            predict_result = self.model.predict(X, batch_size=self.batch_size)
            max_index = np.argmax(predict_result, axis=1)
            return max_index
        else:
            return self.model.predict(X, batch_size=self.batch_size)

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        # First, reshape X to meet LSTM input requirements
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return self.model.predict(X, batch_size=self.batch_size)

    def create_model(self, time_window_size, metric):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(time_window_size, 1)))
        model.add(GlobalMaxPool1D())

        model.add(Dense(units=time_window_size, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
        print(model.summary())
        return model

    def fit(self, X, y):
        self.time_window_size = X.shape[1]

        input_timeseries_dataset = np.expand_dims(X[:, :, 0], axis=2)

        self.model = self.create_model(self.time_window_size, metric=self.metric)
        history = self.model.fit(x=input_timeseries_dataset, y=y,
                                 batch_size=self.batch_size, epochs=self.epochs,
                                 verbose=self.VERBOSE)
        scores = self.predict(dataset)
        scores.sort()
        cut_point = int(self.estimated_negative_sample_ratio * len(scores))
        self.threshold = scores[cut_point]

        print('estimated threshold is ' + str(self.threshold))

        self.config = dict()
        self.config['time_window_size'] = self.time_window_size
        self.config['metric'] = self.metric
        self.config['threshold'] = self.threshold

        return self

    def predict(self, timeseries_dataset):
        input_timeseries_dataset = np.expand_dims(timeseries_dataset, axis=2)
        target_timeseries_dataset = self.model.predict(x=input_timeseries_dataset)
        dist = np.linalg.norm(timeseries_dataset - target_timeseries_dataset, axis=-1)
        return dist

    @staticmethod
    def define_optimizer(optimizer_type='Adam', lr=0.001):
        # Todo: use kwargs to allow for additional optimizer tweaking
        try:
            optimizer_class = getattr(keras.optimizers, optimizer_type)
            optimizer = optimizer_class(lr=lr)
        except AttributeError as ae:
            raise ValueError('Could not find optimizer:',
                             optimizer_type, ' - check spelling!')

        return optimizer

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot