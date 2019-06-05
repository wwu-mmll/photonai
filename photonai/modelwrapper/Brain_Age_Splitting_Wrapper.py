from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralBiclustering
from skimage.util.shape import view_as_windows
import math
import numpy as np
import os
#from photonai.photonlogger.Logger import Logger

class Brain_Age_Splitting_Wrapper(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, patch_size = 25, random_state=42, logs=''):
        self.patch_size = patch_size
        self.random_state = random_state
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y=None):
        pass

    def transform(self, X):

        BenisLänge = X.shape[1] - (self.patch_size - 1)
        BenisBreite = X.shape[2]- (self.patch_size - 1)

        BenisLängsSchritte = BenisLänge / self.patch_size
        BenisBreitenSchritte = BenisBreite / self.patch_size

        KleineBenisLängsSchritte = int(np.ceil(BenisLängsSchritte))
        KleineBenisBreitenSchritte = int(np.ceil(BenisBreitenSchritte))
        BenisSteppos = KleineBenisLängsSchritte * KleineBenisBreitenSchritte

        MegaBenis = BenisSteppos * X.shape[3]
        Beniswertos = np.ones((MegaBenis, self.patch_size, self.patch_size, 1))
        print(Beniswertos.shape)

        for i in range(X.shape[0]):
            # print(brain_scans[i,:,:,:].shape)
            # make brain scans a new variable or try making the statement for the ith one
            # x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).swapaxes(1, 2).reshape(-1, 2, 2)
            Benis = view_as_windows(X[i, :, :, :], (self.patch_size, self.patch_size, 1), step=1)
            #print(Benis.shape)

            BenisLänge = Benis.shape[0]
            BenisBreite = Benis.shape[1]
            #BenisSchritte = BenisLänge / self.patch_size

            BenisMatrix = Benis[0:BenisLänge:self.patch_size, 0:BenisBreite:self.patch_size, :, :]
            #print(BenisMatrix.shape)

            # TODO: Reshape First 3 Matrix Dimensions into 1, which will give 900 images
            BenisMatrix = BenisMatrix.reshape((-1, BenisMatrix.shape[3], BenisMatrix.shape[4], BenisMatrix.shape[5]))
            #print(BenisMatrix.shape)

            Beniswertos = np.append(Beniswertos, BenisMatrix, axis=3)
            #print(Beniswertos.shape)

        #TODO: Drop first row
        Beniswertos = np.delete(Beniswertos, 0, 3)
        Beniswertos = np.moveaxis(Beniswertos, 3, 0)
        print(Beniswertos.shape)

        return Beniswertos

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
