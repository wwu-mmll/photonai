from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import numpy as np


class PhotonOneClassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel='rbf', nu=0.5):
        self.kernel = kernel
        self.nu = nu
        self.my_svm = None

    def fit(self, X, y=None):
        self.my_svm = OneClassSVM(kernel=self.kernel, nu=self.nu)
        self.my_svm.fit(X[y==1])
        return self

    def predict(self, X):
        return self.my_svm.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)

    def set_params(self, **params):
        if 'kernel' in params:
            self.kernel = params['kernel']
        if 'nu' in params:
            self.nu = params['nu']

        if not self.my_svm:
            self.my_svm = OneClassSVM(kernel=self.kernel, nu=self.nu)
        self.my_svm.set_params(**params)

    def get_params(self, deep=True):
        if not self.my_svm:
            self.my_svm = OneClassSVM(kernel=self.kernel, nu=self.nu)
        svm_dict = self.my_svm.get_params(deep)
        return svm_dict
