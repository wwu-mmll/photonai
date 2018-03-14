
from Clustering.ClusterPerformance import ClusterPerformance
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

cluster_metric_helper = ClusterPerformance(KMeans, [2, 3, 4, 5, 6], plot=True)

X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility


cluster_metric_helper.fit(StandardScaler().fit_transform(X))
