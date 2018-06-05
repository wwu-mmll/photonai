from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error as mae
from hashlib import sha1
from pathlib import Path
import numpy as np
from sklearn.externals import joblib
import os
from ..photonlogger.Logger import Logger

class PhotonPCA(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, n_components=None, logs=''):
        self.n_components = n_components
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()
        self.pca = None

    def fit(self, X, y=None):
        hash = sha1(np.asarray(X, order='C')).hexdigest()
        hash_file = Path(str(self.logs + '/' + hash + '_' + str(self.n_components) + '.pkl'))
        if hash_file.is_file():
            Logger().debug('Reloading PCA...')
            self.pca = joblib.load(str(self.logs + '/' + hash + '_' + str(self.n_components) + '.pkl'))
        else:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X)
            joblib.dump(self.pca, self.logs + '/' + hash + '_' + str(self.n_components) +'.pkl')
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def set_params(self, **params):
        if 'n_components' in params:
            self.n_components = params['n_components']
        if 'logs' in params:
            self.logs = params.pop('logs', None)

        if not self.pca:
            self.pca = PCA(n_components=self.n_components)
        self.pca.set_params(**params)

    def get_params(self, deep=True):
        if not self.pca:
            self.pca = PCA(n_components=self.n_components)
        pca_dict = self.pca.get_params(deep)
        pca_dict['logs'] = self.logs
        return pca_dict
