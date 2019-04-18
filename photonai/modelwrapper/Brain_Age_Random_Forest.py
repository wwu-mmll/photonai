import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

class Brain_Age_Random_Forest(BaseEstimator, ClassifierMixin):
    # todo: BUGFIX --> pooling doesnt work
    def __init__(self, target_dimension=2,
                 loss='mse', metrics=['accuracy'],
                 gpu_device='/gpu:0', random_state = 42,
                 early_stopping_flag=True, eaSt_patience=20,
                 reLe_factor=0.4, reLe_patience=5):

        self.target_dimension = target_dimension
        self.loss = loss
        self.metrics = metrics
        self.random_state = random_state
        self.gpu_device = gpu_device
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience

        self.x = None
        self.y_ = None
        self.model = None

    def fit(self, X, y):
        # Reshape X to add dimension for CNN (RGB channel)
        print("Starting Fitting")
        y = np.repeat(y, X.shape[1])
        #make patches per person a training case
        print(X.shape)
        X = np.reshape(X, (-1, X.shape[2], X.shape[3]))
        #flatten training cases for reshape
        X = np.reshape(X, (X.shape[0]), -1)
        #shuffle the the data so there won't be long strands of same-aged people
        X, y = shuffle(X, y, random_state = self.random_state)

        #model is a random forest regressor
        self.photon_rfr = RandomForestRegressor()
        self.photon_rfr.fit[X, y]
        print("Fitting done")

        return self

    def predict(self, X):
            return self.photon_rfr.predict(X)


    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot