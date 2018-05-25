import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def plot_some_data(data, targets):
    ax_array = np.arange(0, data.shape[0], 1)
    plt.figure().clear()
    plt.plot(ax_array, data, ax_array, targets)
    plt.title('A sample of data')
    plt.show()


def show_pca(X, y):
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig)

    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(X)
    # print('3 PCs:', np.sum(pca.explained_variance_ratio_[0:2]))
    print('PCA 3D explained_variance', np.sum(pca.explained_variance_ratio_))
    X = pca.transform(X)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)  # cmap=plt.cm.spectral

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
