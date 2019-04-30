from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralBiclustering
import numpy as np
import os
from matplotlib import pyplot as plt
#from photonai.photonlogger.Logger import Logger

class Biclustering2d(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, n_clusters=4, random_state=42, scale='bistochastic', n_components=6,
                 n_best=3, logs=''):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale  # ‘scale’, ‘bistochastic’, or ‘log’ (log cannot handle sparse data)
        self.n_components = n_components
        self.n_best = n_best
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        # Biclustering of the mean 2d image of all samples
        HC = X[y==0,:]
        print(HC.shape)
        X_mean = np.squeeze(np.mean(HC, axis=0))
        #X_mean = np.squeeze(np.mean(X, axis=0))
        self.biclustModel = self.create_model()
        self.biclustModel.fit(X_mean)

        # Plotting the clustered matrix
        fit_data = X_mean[np.argsort(self.biclustModel.row_labels_)]
        fit_data = fit_data[:, np.argsort(self.biclustModel.column_labels_)]
        plt.matshow(fit_data)
        plt.title(self.n_clusters)
        plt.show()

        return self

    def transform(self, X):
        X_reordered = np.empty(X.shape)
        for i in range(X.shape[0]):
            x = np.squeeze(X[i,:,:])
            x_clust = x[np.argsort(self.biclustModel.row_labels_)]
            x_clust = x_clust[:, np.argsort(self.biclustModel.column_labels_)]
            X_reordered[i, :, :] = x_clust
        return X_reordered

    def create_model(self):

        biclustModel = SpectralBiclustering(n_clusters=self.n_clusters, random_state=self.random_state,
                                            method=self.scale, n_components=self.n_components,
                                            n_best=self.n_best)



        return biclustModel

    # ToDo: add these functions again
    # def set_params(self, **params):
    #     if 'n_components' in params:
    #         self.n_clusters = params['n_components']
    #     if 'logs' in params:
    #         self.logs = params.pop('logs', None)
    #
    #     if not self.biclustModel:
    #         self.biclustModel = self.createBiclustering()
    #     self.biclustModel.set_params(**params)
    #
    # def get_params(self, deep=True):
    #     if not self.biclustModel:
    #         self.biclustModel = self.createBiclustering()
    #     biclust_dict = self.biclustModel.get_params(deep)
    #     biclust_dict['logs'] = self.logs
    #     return biclust_dict

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     X = np.random.rand(100, 30, 30)
#     bcm = Biclustering2d(n_clusters=3)
#     bcm.fit(X)
#     X_new = bcm.transform(X)
#     plt.matshow(np.squeeze(X_new[0]), cmap=plt.cm.Blues)
#     plt.show()
