# DUMMY IMPLEMENTATION

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_absolute_error as mae


class PCA_AE_Wrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=5):

        print(n_components)
        self.n_components = n_components
        # self.n_components = None
        self.X_recon = None
        self.my_pca = None

    def fit(self, X, y=None):
        self.my_pca = PCA(n_components=self.n_components)
        print(self.my_pca)
        self.my_pca.fit(X)
        return self

    def predict(self, X):
        tmp = self.my_pca.transform(X)
        self.X_recon = self.my_pca.inverse_transform(tmp)
        # return self.X_recon
        return self.transform(X)
        # return X[:, 0:self.n_components]

    def transform(self, X):
        return self.my_pca.transform(X)

    def score(self, X, y=None):
        # y=None
        self.predict(X)
        loss = mae(X, self.X_recon)
        return loss