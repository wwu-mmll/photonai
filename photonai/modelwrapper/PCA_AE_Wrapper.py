# DUMMY IMPLEMENTATION

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error as mae


class PCA_AE_Wrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "transformer"

    def __init__(self, n_components=5):
        self.n_components = n_components
        self.my_pca = None

    def fit(self, X, y=None):
        self.my_pca = PCA(n_components=self.n_components)
        self.my_pca.fit(X)
        return self

    def transform(self, X):
        return self.my_pca.transform(X)

    def score(self, X, y=None):
        # y=None
        X_transformed = self.transform(X)
        X_recon = self.my_pca.inverse_transform(X_transformed)
        loss = mae(X, X_recon)
        return loss