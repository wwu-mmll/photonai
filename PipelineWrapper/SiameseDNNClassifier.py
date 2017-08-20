
import time
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Input, Lambda
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from Framework.Metrics import categorical_accuracy_score
from keras import backend as K

class SiameseDNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, input_dim, target_dimension = 10, dropout_rate=0.5, act_func='relu',
                 learning_rate=0.1, batch_normalization=True, nb_epoch=10000, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5):

        self.target_dimension = target_dimension
        self.dropout_rate = dropout_rate
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.batch_normalization = batch_normalization
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.input_dim = input_dim
        self.model = None

    def fit(self, X, y):

        # 1. make model
        self.model = self.create_model()

        # 2. fit model
        # start_time = time.time()

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
                                          min_lr=0.001, verbose=1)
            callbacks_list += [reduce_lr]

            # create training+test positive and negative pairs
            digit_indices = [np.where(y_train == i)[0] for i in range(10)]
            tr_pairs, tr_y = self.create_pairs(X_train, digit_indices)

            digit_indices = [np.where(y_val == i)[0] for i in range(10)]
            te_pairs, te_y = self.create_pairs(X_val, digit_indices)

            # fit the model
            results = self.model.fit(tr_pairs, tr_y,
                                     validation_data=(te_pairs, te_y),
                                     batch_size=128,
                                     epochs=self.nb_epoch,
                                     verbose=0,
                                     callbacks=callbacks_list)

            # Use learnt features for classification
            # prepare target values
            # Todo: calculate number of classes?
            try:
                if (self.target_dimension > 1) and (y.shape[1] > 1):
                    y = self.dense_to_one_hot(y, self.target_dimension)
            except:
                pass

            model = self.create_base_network()


        else:
            pass

        return self

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=128)
        max_index = np.argmax(predict_result, axis=1)
        return self.dense_to_one_hot(max_index, self.target_dimension)

    def score(self, X, y_true):
        return np.zeros(1)

    def create_model(self):
        # network definition
        base_network = self.create_base_network(self.input_dim)

        input_a = Input(shape=(self.input_dim,))
        input_b = Input(shape=(self.input_dim,))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss=self.contrastive_loss, optimizer=optimizer)

        return model

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    def create_pairs(self, x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def create_base_network(self):
        '''Base network to be shared (eq. to feature extraction).
        '''
        seq = Sequential()
        seq.add(Dense(64, input_shape=(self.input_dim,), kernel_initializer='random_uniform'))
        seq.add(BatchNormalization())
        seq.add(Activation(self.act_func))
        seq.add(Dropout(self.dropout_rate))
        seq.add(Dense(32), kernel_initializer='random_uniform')
        seq.add(BatchNormalization())
        seq.add(Activation(self.act_func))
        return seq

    def compute_accuracy(self, predictions, labels):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''

        return labels[predictions.ravel() < 0.5].mean()

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot