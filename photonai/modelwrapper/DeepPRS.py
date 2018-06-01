#========================================#
#      Wrapper for DeepPRS Model         #
#========================================#

# Nils Winter
# University of Münster
# Translational Psychiatry
# nils.r.winter@gmail.com
# 8th May 2018

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense, Lambda
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from ..logging.Logger import Logger
from ..helpers.TFUtilities import binary_to_one_hot


class DeepPRSClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_nodes=1, weights=None, add_noise=False, dropout_rate=0.5, learning_rate=0.1,
                 nb_epoch=10000, early_stopping_flag=True,  eaSt_patience=20, reLe_factor = 0.4,
                 reLe_patience=5, batch_size=64):

        self.n_nodes = n_nodes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.batch_size = batch_size
        self.weights = weights
        self.bias = None
        self.add_noise = add_noise
        self.model = None

        if Logger().verbosity_level == 2:
            self.verbosity = 2
        else:
            self.verbosity = 0

    def fit(self, X, y):

        # prepare target values
        try:
            if y.shape[1] < 2:
                y = binary_to_one_hot(y)
        except:
            y = binary_to_one_hot(y)

        input_size = X.shape[1]

        # 1. prepare weights
        b = np.zeros(self.n_nodes)
        W = np.tile(self.weights, (1, self.n_nodes))
        if self.add_noise:
            mu, sigma = 0, 0.01
            # creating a noise with the same dimension as the dataset (2,2)
            noise = np.random.normal(mu, sigma, [input_size, self.n_nodes])
            W += noise

        # 2. make model

        self.model = self.create_model(input_size, W, b)

        # 3. fit model
        # use callbacks only when size of training set is above 100
        if X.shape[0] > 100:
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
            self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity,
                                     callbacks=callbacks_list)
        else:
            # fit the model
            Logger().warn('Cannot use Keras Callbacks. Not enough samples...')
            self.model.fit(X, y, batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity)

        return self

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=self.batch_size)
        max_index = np.argmax(predict_result, axis=1)
        return max_index

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        return self.model.predict(X, batch_size=self.batch_size)

    def create_model(self, input_size, W, b):

        model = Sequential()
        model.add(Dropout(self.dropout_rate, input_shape=(input_size,)))
        model.add(Dense(self.n_nodes, activation='linear', weights=np.asarray([W, b])))
        model.add(Lambda(lambda x: tf.divide(x, W.shape[0])))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model

    def get_prs(self, X):
        model = Model(input=self.model.layers[0], output=[self.model.layers[3]])
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model.predict(X)


