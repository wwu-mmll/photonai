import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import shuffle
from photonai.modelwrapper.KerasDNNRegressor import KerasDNNRegressor


class BrainAgeNeuralNet(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.estimator = KerasDNNRegressor(**kwargs)

    def fit(self, X, y):

        y = np.repeat(y, X.shape[1])
        # make patches per person a training case
        print(X.shape)
        X = np.reshape(X, (-1, X.shape[2], X.shape[3]))
        # flatten training cases for reshape
        X = np.reshape(X, (X.shape[0]), -1)
        # shuffle the the data so there won't be long strands of same-aged people
        X, y = shuffle(X, y, random_state=self.random_state)

        # 1. make model
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            print("Loading data")
            X = np.asarray(X)

        X_to_predict = X.reshape(X.shape[0], X.shape[1], -1)
        predict_result = []
        for i in range(X.shape[0]):
            predict_interim_result = np.squeeze(self.estimator.predict(X_to_predict[i, :, :], batch_size=self.batch_size))
            # predict_interim_result = self.photon_rfr.predict(X_to_predict[i, :, :])
            predict_result_to_append = np.mean(predict_interim_result)
            predict_result.append(predict_result_to_append)
        return predict_result



