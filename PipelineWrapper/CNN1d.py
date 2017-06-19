import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin

class CNN1d(BaseEstimator, ClassifierMixin):

    def __init__(self, image_width=24*12, image_height=5*60, num_labels=5, learning_rate=1e-4,
                 number_densely_neurons=1024, patch_size=100, num_features=32, reduction_1=6, reduction_2=2):

        self.image_width = image_width
        self.image_height = image_height
        self.num_labels = num_labels

        self.learning_rate = learning_rate

        self.number_densely_neurons = number_densely_neurons
        self.patch_size = patch_size
        self.num_features = num_features
        self.reduction_1 = reduction_1
        self.reduction_2 = reduction_2

        self.x = None
        self.y_ = None
        self.y_conv = None

    def create_model(input_shape=[], n_classes=[], n_filters=[],
                           kernel_sizes=[], n_convolutions_per_block=[],
                           pooling_size=[],
                           strides=[], size_last_layer=[],
                           actFunc='relu', learning_rate=0.001,
                           dropout_rate=0, batch_normalization=False,
                           nb_epoch=200,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'], optimizer='adam',
                           gpu_device='/gpu:0'):

        # size_last_layer
        # n_filters
        # kernel_size
        # strides
        # n_convolutions_per_block

        ks = kernel_sizes
        ps = pooling_size
        nf = n_filters
        input_shape = (input_shape[0], input_shape[1], input_shape[2], 1)

        model = Sequential()

        for ind_blocks in range(len(n_filters)):
            for ind_convs in range(n_convolutions_per_block):
                if ind_blocks == 0 and ind_convs == 0:
                    with tf.device(gpu_device):
                        model.add(Conv3D(nf[ind_blocks],
                                         [ks[0], ks[1], ks[2]],
                                         strides=strides,
                                         padding='same',
                                         input_shape=input_shape))
                        model.add(Activation(actFunc))
                    if batch_normalization:
                        model.add(BatchNormalization())
                else:
                    with tf.device(gpu_device):
                        model.add(Conv3D(nf[ind_blocks],
                                         [ks[0], ks[1], ks[2]],
                                         strides=strides,
                                         padding='same'))
                        model.add(Activation(actFunc))

                    if batch_normalization:
                        model.add(BatchNormalization())
            with tf.device(gpu_device):
                if ps:
                    model.add(MaxPooling3D(pool_size=(ps, ps, ps)))

                if dropout_rate:
                    model.add(Dropout(dropout_rate))

        with tf.device(gpu_device):
            model.add(Flatten())
            model.add(Dense(size_last_layer))
            model.add(Activation(actFunc))
            if dropout_rate:
                model.add(Dropout(dropout_rate))
        if batch_normalization:
            model.add(BatchNormalization())
        with tf.device(gpu_device):
            model.add(Dense(n_classes))
            model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        optimizer = define_optimizer(optimizer_type=optimizer,
                                     lr=learning_rate)

        # Let's train the model using RMSprop
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print(model.summary())
        return model

    def define_optimizer(optimizer_type='adam', lr=0.001):
        # TO DO: - use *kwargs to allow specification of additional optimizer hyperparameters
        #        - learn optimizer, see DeepMind paper
        #        - enable tf custom optimizer use
        if optimizer_type.lower() == 'adam':
            optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9,
                                              beta_2=0.999,
                                              epsilon=1e-08, decay=0.0)

        elif optimizer_type.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0,
                                             decay=0.0, nesterov=False)

        elif optimizer_type.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9,
                                                 epsilon=1e-08,
                                                 decay=0.0)

        elif optimizer_type.lower() == 'adagrad':
            optimizer = keras.optimizers.Adagrad(lr=lr, epsilon=1e-08,
                                                 decay=0.0)

        elif optimizer_type.lower() == 'adadelta':
            optimizer = keras.optimizers.Adadelta(lr=lr, rho=0.95,
                                                  epsilon=1e-08,
                                                  decay=0.0)

        elif optimizer_type.lower() == 'adamax':
            optimizer = keras.optimizers.Adamax(lr=lr, beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-08,
                                                decay=0.0)

        elif optimizer_type.lower() == 'nadam':
            optimizer = keras.optimizers.Nadam(lr=lr, beta_1=0.9,
                                               beta_2=0.999,
                                               epsilon=1e-08,
                                               schedule_decay=0.004)

        else:
            raise ValueError('Choose a valid keras optimizer!')

        return optimizer