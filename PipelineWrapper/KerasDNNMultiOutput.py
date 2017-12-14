import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from Logging.Logger import Logger
from Framework.Metrics import variance_explained_score
from Framework.Validation import Scorer

class KerasDNNMultiOutput(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_layer_sizes=[10, 20], list_of_outputs=[],
                 dropout_rate=0.5, target_dimension=10,
                 act_func='prelu', learning_rate=0.1, batch_normalization=True,
                 nb_epoch=100, early_stopping_flag=True, batch_size=32,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5,
                 scoring_method='variance_explained'):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.target_dimension = target_dimension
        self.batch_normalization = batch_normalization
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.batch_size = batch_size
        self.list_of_outputs = list_of_outputs
        self.scoring_method = Scorer.create(scoring_method)

        self.model = None

        if Logger().verbosity_level == 2:
            self.verbosity = 2
        else:
            self.verbosity = 0

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        multi_y = []
        for i in range(y.shape[1]):
            multi_y.append(y[:,i])

        self.model = self.create_model()

        # use callbacks only when size of training set is above 100
        if X.shape[0] > 100:

            # get pseudo validation set for keras callbacks
            splitter = ShuffleSplit(n_splits=1, test_size=0.2)
            for train_index, val_index in splitter.split(X):
                multi_y = []
                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]

                multi_y_train = []
                multi_y_val = []
                for i in range(y.shape[1]):
                    multi_y_train.append(y_train[:, i])
                    multi_y_val.append(y_val[:, i])

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
            results = self.model.fit(X_train, multi_y_train,
                                     validation_data=(X_val, multi_y_val),
                                     batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity,
                                     callbacks=callbacks_list)
        else:
            multi_y = []
            for i in range(y.shape[1]):
                multi_y.append(y[:, i])

            # fit the model
            Logger().warn('Cannot use Keras Callbacks because of small sample size...')
            results = self.model.fit(X, multi_y, batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity)

        return self

    def predict(self, X):
        preds = np.transpose(np.squeeze(np.asarray(self.model.predict(X, batch_size=self.batch_size))))
        return preds

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        return self.model.predict(X, batch_size=self.batch_size)

    def score(self, X, y_true):
        preds = self.predict(X)
        scores = []
        for i in range(preds.shape[1]):
            scores.append(self.scoring_method(y_true[:,i],preds[:,i]))
        return scores

    def create_model(self):

        input_layer = Input(shape=(self.input_dim,), dtype='float32', name='input_layer')

        for i, dim in enumerate(self.hidden_layer_sizes):
            if i == 0:
                x = Dense(dim,  kernel_initializer='random_uniform')(input_layer)
            else:
                x = Dense(dim, kernel_initializer='random_uniform')(x)

            if self.batch_normalization == 1:
                x = BatchNormalization()(x)

            if self.act_func == 'prelu':
                x = PReLU(alpha_initializer='zero', weights=None)(x)
            else:
                x = Activation(self.act_func)(x)

            x = Dropout(self.dropout_rate)(x)

        # define all output nodes
        outputs = []
        losses = []
        loss_weights = []
        for output_node in self.list_of_outputs:
            outputs.append(Dense(output_node['target_dimension'],
                                                 activation=output_node['activation'])(x))
            losses.append(output_node['loss'])
            loss_weights.append(output_node['loss_weight'])


        model = Model(input_layer,outputs)
        #model.summary()
        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss=losses, loss_weights=loss_weights,
                      optimizer=optimizer)


        return model

