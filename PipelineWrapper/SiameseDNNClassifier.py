import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense, Input, Lambda
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit

from Helpers.TFUtilities import oneHot


class SiameseDNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, input_dim=10, n_pairs_per_sample =2 , target_dimension = 10, dropout_rate=0.5, act_func='relu',
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
        self.n_pairs_per_sample = n_pairs_per_sample
        self.model = None

    def fit(self, X, y):

        # 1. make model
        siamese_model = self.create_model()

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
                                          min_lr=0.001, verbose=0)
            callbacks_list += [reduce_lr]

            # create training+test positive and negative pairs
            y_train_oh_reverse = oneHot(y_train,reverse=True)
            y_val_oh_reverse = oneHot(y_val,reverse=True)
            
            digit_indices = [np.where(y_train_oh_reverse == i)[0] for i in range(10)]
            tr_pairs, tr_y = self.create_pairs(X_train, digit_indices, self.n_pairs_per_sample)

            digit_indices = [np.where(y_val_oh_reverse == i)[0] for i in range(10)]
            te_pairs, te_y = self.create_pairs(X_val, digit_indices, self.n_pairs_per_sample)
            
            print(tr_pairs.shape)
            print(te_pairs.shape)
            # fit the model
            results = siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                         validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
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
            
            seq_model = siamese_model.get_layer(index=2)
            seq_model.layers[3] = Dropout(0.5)
            new_input = Input(shape=(self.input_dim,))
            new_seq = seq_model(new_input)
            new_seq = Dropout(0.5)(new_seq)
            new_seq = Dense(self.target_dimension, activation='softmax')(new_seq)
            self.model = Model(new_input, new_seq)
            optimizer = Adam(lr=self.learning_rate)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            print(self.model.summary())
            results = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=128,
                                     epochs=self.nb_epoch,
                                     verbose=0,
                                     callbacks=callbacks_list)

        else:
            pass

        return self

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=128)
        max_index = np.argmax(predict_result, axis=1)
        return self.dense_to_one_hot(max_index, self.target_dimension)

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=128)

    def score(self, X, y_true):
        return np.zeros(1)

    def create_model(self):
        # network definition
        base_network = self.create_base_network()

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

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    @staticmethod
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        """Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    # def create_pairs(self, x, digit_indices):
    #     '''Positive and negative pair creation.
    #     Alternates between positive and negative pairs.
    #     '''
    #     pairs = []
    #     labels = []
    #     n = min([len(digit_indices[d]) for d in range(10)]) - 1
    #     for d in range(10):
    #         for i in range(n):
    #             z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
    #             pairs += [[x[z1], x[z2]]]
    #             inc = random.randrange(1, 10)
    #             dn = (d + inc) % 10
    #             z1, z2 = digit_indices[d][i], digit_indices[dn][i]
    #             pairs += [[x[z1], x[z2]]]
    #             labels += [1, 0]
    #     return np.array(pairs), np.array(labels)

    def create_base_network(self):
        """Base network to be shared (eq. to feature extraction).
        """
        seq = Sequential()
        seq.add(Dense(128, input_shape=(self.input_dim,), kernel_initializer='random_uniform'))
        seq.add(BatchNormalization())
        seq.add(Activation(self.act_func))
        seq.add(Dropout(0.1))
        seq.add(Dense(64, kernel_initializer='random_uniform'))
        seq.add(BatchNormalization())
        seq.add(Activation(self.act_func))
        return seq

    @staticmethod
    def compute_accuracy(predictions, labels):
        """Compute classification accuracy with a fixed threshold on distances.
        """

        return labels[predictions.ravel() < 0.5].mean()

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def create_pairs(self, x, class_indices, n_pairs_per_subject):
        """Positive and negative pair creation.
        Alternates between positive and negative pairs.
        """
        # x: data, class_indices: lists of indices of subjects in classes

        n_sample_pairs = 2 * n_pairs_per_subject * len(class_indices[0]) * len(class_indices)

        # if n_sample_pairs > 100000:
        #     print('INSANE: You are trying to use ', n_sample_pairs,
        #           'sample pairs.')

        print('Generating ', n_sample_pairs, 'sample pairs.')

        pairs = []
        labels = []
        n = len(class_indices[0])
        if all(len(s) == n for s in class_indices):
            raise ValueError('Lists do not have the same length.')

        pos_pairs = self.draw_pos_pairs(class_indices, n_pairs_per_subject)
        neg_pairs = self.draw_neg_pairs(class_indices, n_pairs_per_subject)

        for d in range(len(pos_pairs)):
            z1, z2 = pos_pairs[d]
            pairs += [[x[z1], x[z2]]]
            z1, z2 = neg_pairs[d]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
        return np.array(pairs), np.array(labels)

    @staticmethod
    def draw_pos_pairs(indices, n_pairs_per_subject):
        pairs = []
        for ind_lists in range(len(indices)):
            a = indices[ind_lists]
            for ind_pair in range(n_pairs_per_subject):
                for ind_sub in range(len(a)):
                    p1 = a[ind_sub]
                    next_ind = (ind_sub + 1 + ind_pair) % len(a)
                    p2 = a[next_ind]
                    pairs.append([p1, p2])
        return pairs

    @staticmethod
    def draw_neg_pairs(indices, n_pairs_per_subject):
        pairs = []
        n_classes = len(indices)
        n_subs = len(indices[0])
        for ind_lists in range(n_classes):
            for ind_pair in range(n_pairs_per_subject):
                for ind_sub in range(len(indices[ind_lists])):
                    p1 = indices[ind_lists][ind_sub]
                    next_ind = (ind_sub + ind_pair + (ind_lists * (
                        n_pairs_per_subject - 1)) + ind_lists) % len(indices[(ind_lists+ind_pair)%n_classes])
                    p2 = indices[(ind_lists+ind_pair)%n_classes][next_ind]
                    pairs.append([p1, p2])
        return pairs