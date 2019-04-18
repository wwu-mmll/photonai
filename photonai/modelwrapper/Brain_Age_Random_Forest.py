import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

class Brain_Age_Random_Forest(BaseEstimator, ClassifierMixin):
    # todo: BUGFIX --> pooling doesnt work
    def __init__(self):
        pass

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
