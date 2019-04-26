import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.utils import shuffle


class Brain_Age_Random_Forest(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"

    # todo: BUGFIX --> pooling doesnt work
    def __init__(self):
        pass

    def fit(self, X, y):

        print("hello here is random forest ")

        if not isinstance(X, np.ndarray):
            print("converting data to numpy array")
            X = np.asarray(X)

        # Reshape X to add dimension for CNN (RGB channel)
        print("Starting Fitting")
        y = np.repeat(y, X.shape[1])
        # make patches per person a training case
        print(X.shape)
        X = np.reshape(X, (-1, X.shape[2], X.shape[3]))
        # flatten training cases for reshape
        X = np.reshape(X, (X.shape[0], -1))
        # shuffle the the data so there won't be long strands of same-aged people
        X, y = shuffle(X, y, random_state=42)

        # model is a random forest regressor
        # RandomForestRegressor()
        self.photon_rfr = LinearSVR()
        self.photon_rfr.fit(X, y)
        print("Fitting done")

        return self

    def predict(self, X):

        print("hello here is random forest ")

        if not isinstance(X, np.ndarray):
            print("converting data to numpy array")
            X = np.asarray(X)

        X = np.reshape(X, (-1, X.shape[2], X.shape[3]))
        # flatten training cases for reshape
        X = np.reshape(X, (X.shape[0], -1))
        X = shuffle(X, random_state=42)
        print("now predicting")
        return self.photon_rfr.predict(X)
