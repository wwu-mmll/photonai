from sklearn.base import BaseEstimator
from sklearn.preprocessing import OrdinalEncoder
from photonai.helper.helper import PhotonDataHelper
import numpy as np


class FeatureEncoder(BaseEstimator):

    def __init__(self):
        self.encoder_list = []

    def fit(self, X, y=None, **kwargs):
        # iterate X and if its a string column: encode and save encoder
        self.encoder_list = list()
        for i in range(X.shape[1]):
            feature = X[:, i]
            if isinstance(feature[0], str):
                new_encoder = OrdinalEncoder()
                feature = np.reshape(feature, (-1, 1))
                new_encoder.fit(feature)
                self.encoder_list.append(new_encoder)
            else:
                self.encoder_list.append(None)

    def transform(self, X, y=None, **kwargs):
        # iterate X and apply encoder if necessary
        new_X = None
        for i in range(X.shape[1]):
            feature = X[:, i]
            transformer = self.encoder_list[i]
            if transformer is not None:
                feature = np.reshape(feature, (-1, 1))
                trans_X = transformer.transform(feature)
            else:
                trans_X = feature
            new_X = PhotonDataHelper.stack_data_horizontally(new_X, trans_X)
        return new_X

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X, y=None, **kwargs):
        new_X = None
        for i in range(X.shape[1]):
            feature = X[:, i]
            transformer = self.encoder_list[i]
            if transformer is not None:
                feature = np.reshape(feature, (-1, 1))
                trans_X = transformer.inverse_transform(feature)
            else:
                trans_X = feature
            new_X = PhotonDataHelper.stack_data_horizontally(new_X, trans_X)
        return new_X
