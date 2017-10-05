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

class SimpleLSTM(BaseEstimator, ClassifierMixin):
    # todo: BUGFIX --> pooling doesnt work
    def __init__(self, target_dimension=2, units=32, size_last_layer=10,
                 gaussian_noise=0, act_func='relu', learning_rate=0.001,
                 dropout_rate=0, batch_normalization=True,
                 nb_epochs=200, batch_size=64,
                 loss='categorical_crossentropy', metrics=['accuracy'],
                 optimizer='adam', gpu_device='/gpu:0',
                 early_stopping_flag=True, eaSt_patience=20,
                 reLe_factor=0.4, reLe_patience=5):

        self.target_dimension = target_dimension
        self.units = units
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
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience

        self.x = None
        self.y_ = None
        self.model = None

    def fit(self, X, y):
        if self.target_dimension > 1:
            y = self.dense_to_one_hot(y, self.target_dimension)

        self.model = self.create_model(X.shape)

        # Reshape X to add dimension for CNN (RGB channel)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # use callbacks only when size of training set is above 100
        if X.shape[-1] > 100:
            # get pseudo validation set for keras callbacks
            splitter = ShuffleSplit(n_splits=1, test_size=0.2)
            for train_index, val_index in splitter.split(X):
                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]

            # register callbacks
            callbacks_list = []
            # use early stopping (to save time;
            # does not improve performance as checkpoint will find the best model anyway)
            if self.early_stopping_flag:
                early_stopping = EarlyStopping(monitor='val_loss',
                                               patience=self.eaSt_patience)
                callbacks_list += [early_stopping]

            # adjust learning rate when not improving for patience epochs
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                          factor=self.reLe_factor,
                                          patience=self.reLe_patience,
                                          min_lr=0.001, verbose=0)
            callbacks_list += [reduce_lr]

            # fit the model
            results = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=128,
                                     epochs=self.nb_epochs,
                                     verbose=0,
                                     callbacks=callbacks_list)
        else:
            # fit the model
            print(
                'Cannot use Keras Callbacks because of small sample size...')
            results = self.model.fit(X, y, batch_size=128,
                                     epochs=self.nb_epochs,
                                     verbose=0)

        return self

    def predict(self, X):
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        if self.target_dimension > 1:
            predict_result = self.model.predict(X, batch_size=self.batch_size)
            max_index = np.argmax(predict_result, axis=1)
            return max_index
        else:
            return self.model.predict(X, batch_size=self.batch_size)

    def predict_proba(self, X):
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return self.model.predict(X, batch_size=self.batch_size)

    def create_model(self, input_shape):
        model = Sequential()

        with tf.device(self.gpu_device):
            model.add(LSTM(self.units, input_dim=input_shape[1],
                           stateful=False,return_sequences=False))
            model.add(Activation(self.act_func))
        if self.batch_normalization:
            model.add(BatchNormalization())

        with tf.device(self.gpu_device):
            if self.dropout:
                model.add(Dropout(self.dropout))

        with tf.device(self.gpu_device):
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
        #model.summary()
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