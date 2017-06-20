import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, GaussianNoise
import keras.optimizers
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.base import BaseEstimator, ClassifierMixin

class CNN1d(BaseEstimator, ClassifierMixin):
    # todo: BUGFIX --> pooling doesnt work
    def __init__(self, target_dimension=2, n_filters=[16], kernel_size=3,
                 pooling_size=3, stride=5, size_last_layer=10,
                 n_convolutions_per_block=1, gaussian_noise=0,
                 act_func='relu', learning_rate=0.001,
                 dropout_rate=0, batch_normalization=True,
                 nb_epochs=200, batch_size=64,
                 loss='categorical_crossentropy', metrics=['accuracy'],
                 optimizer='adam', gpu_device='/gpu:0'):

        self.target_dimension = target_dimension
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_convolutions_per_block = n_convolutions_per_block
        self.pooling_size = pooling_size
        self.stride = stride
        self.size_last_layer = size_last_layer
        self.act_func = act_func
        self.lr = learning_rate
        self.dropout = dropout_rate
        self.batch_normalization = batch_normalization
        self.nb_epochs = nb_epochs
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gpu_device = gpu_device
        self.gaussian_noise = gaussian_noise

        self.x = None
        self.y_ = None
        self.model = None

    def fit(self, X, y):
        if self.target_dimension > 1:
            y = self.dense_to_one_hot(y, self.target_dimension)

        self.model = self.create_model(X.shape)
        self.model.fit(X,y, batch_size=self.batch_size, epochs=self.nb_epochs, verbose=0)

    def predict(self, X):
        if self.target_dimension > 1:
            predict_result = self.model.predict(X, batch_size=self.batch_size)
            max_index = np.argmax(predict_result, axis=1)
            return max_index
        else:
            return self.model.predict(X, batch_size=self.batch_size)

    def create_model(self, input_shape):
        model = Sequential()
        input_shape = (input_shape[1], input_shape[2])
        for ind_blocks in range(len(self.n_filters)):
            for ind_convs in range(self.n_convolutions_per_block):
                if ind_blocks == 0 and ind_convs == 0:
                    with tf.device(self.gpu_device):
                        print(input_shape)
                        model.add(Conv1D(self.n_filters[ind_blocks],
                                         self.kernel_size,
                                         strides=self.stride,
                                         padding='same',
                                         input_shape=input_shape))
                        model.add(Activation(self.act_func))
                    if self.batch_normalization:
                        model.add(BatchNormalization())
                else:
                    with tf.device(self.gpu_device):
                        model.add(Conv1D(self.n_filters[ind_blocks],
                                         self.kernel_size,
                                         strides=self.stride,
                                         padding='same'))
                        model.add(Activation(self.act_func))

                    if self.batch_normalization:
                        model.add(BatchNormalization())
            with tf.device(self.gpu_device):
                if self.pooling_size:
                    model.add(MaxPooling1D(pool_size=self.pooling_size))

                if self.dropout:
                    model.add(Dropout(self.dropout))

        with tf.device(self.gpu_device):
            model.add(Flatten())
            if self.gaussian_noise:
                model.add(GaussianNoise(stddev=self.gaussian_noise))
            model.add(Dense(self.size_last_layer))
            model.add(Activation(self.act_func))
            if self.dropout:
                model.add(Dropout(self.dropout))
        if self.batch_normalization:
            model.add(BatchNormalization())
        with tf.device(self.gpu_device):
            model.add(Dense(self.target_dimension))
            model.add(Activation('softmax'))

        optimizer = self.define_optimizer(optimizer_type=self.optimizer,
                                     lr=self.lr)

        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        model.summary()
        return model

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