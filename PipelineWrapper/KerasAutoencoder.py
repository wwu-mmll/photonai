from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
import keras as k
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import backend as K
from sklearn.metrics import mean_squared_error as mae


class SimpleAutoencoder(BaseEstimator, RegressorMixin):

    def __init__(self, n_hidden=10, dropout_rate=0.5):
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.model = None

    def fit(self, X, y):
        n_dim_in = X.shape[1]
        x = Input(shape=(n_dim_in,))
        encoded = Dense(self.n_hidden, activation='relu')(x)
        encoded = Dropout(self.dropout_rate)(encoded)
        decoded = Dense(n_dim_in, activation='sigmoid')(encoded)

        # this model maps an input to its reconstruction
        self.model = Model(x, decoded)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, X, epochs=100, batch_size=16, verbose=0)
        return self

    def transform(self, X):
        get_z = K.function([self.model.layers[0].input], [self.model.layers[1].output])
        return get_z([X])[0]

    def score(self, X, y):
        get_recon = K.function([self.model.layers[0].input], [self.model.layers[2].output])
        decoded = get_recon([X])[0]
        loss = mae(X, decoded)
        return loss


