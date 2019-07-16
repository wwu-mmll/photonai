import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from photonai.modelwrapper.KerasBaseEstimator import KerasBaseEstimator


class PretrainedCNNClassifier(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

    def __init__(self, input_shape=(244,244,3), target_dimension=10,
                 learning_rate=0.001, nb_epoch=10000, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5, freezing_point=0,
                 batch_size=64, ckpt_name='', weight_class_a=1, weight_class_b=1):
        super(KerasBaseEstimator, self).__init__()

        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.target_dimension = target_dimension
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.freezing_point = freezing_point
        self.model = None
        self.batch_size = batch_size
        self.ckpt_name = ckpt_name
        self.weight_class_a = weight_class_a
        self.weight_class_b = weight_class_b


    def fit(self, X, y):

        # prepare target values
        # Todo: calculate number of classes?
        try:
            if (self.target_dimension > 1) and (y.shape[1] > 1):
                y = self.dense_to_one_hot(y, self.target_dimension)
        except:
            pass

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

            if self.ckpt_name:
                # checkpoint to find the model with the best validation performance later (for final testing)
                checkpoint = ModelCheckpoint(self.ckpt_name,
                                             monitor='val_acc',
                                             verbose=0,
                                             save_best_only=True,
                                             mode='auto',
                                             save_weights_only=False)
                callbacks_list += [checkpoint]

            # fit the model
            results = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=1,
                                     callbacks=callbacks_list,
                                     class_weight={0:self.weight_class_a,1:self.weight_class_b})
        else:
            # fit the model
            print('Cannot use Keras Callbacks because of small sample size...')
            results = self.model.fit(X, y, batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=1)

        return self

    def predict(self, X):
        predict_result = self.model.predict(X, batch_size=128)
        return np.argmax(predict_result, axis=1)

    def score(self, X, y_true):
        return np.zeros(1)

    def create_model(self):

        input_tensor = Input(shape=self.input_shape)  # this assumes K.image_data_format() == 'channels_last'

        # create the base pre-trained model
        #base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(self.size_additional_layer, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.target_dimension, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers[:self.freezing_point]:
            layer.trainable = False
            print('Freeze layer ', layer)
        for layer in model.layers[self.freezing_point:]:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        #optimizer = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=0.1, decay=0.9)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        #base_model.summary()
        #model.summary()

        return model

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
