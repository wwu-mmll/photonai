import pandas as pd
from sklearn.cluster import k_means
from scipy.spatial import distance


# nc is number of clusters
# to be implemented without the use of any libraries (from the scratch)

def compute_s(i, x, labels, clusters):
    norm_c = len(clusters)
    s = 0
    for x in clusters:
        # print x
        s += distance.euclidean(x, clusters[i])
    return s


def compute_Rij(i, j, x, labels, clusters, nc):
    Rij = 0
    try:
        # print "h"
        d = distance.euclidean(clusters[i], clusters[j])
        # print d
        Rij = (compute_s(i, x, labels, clusters) + compute_s(j, x, labels, clusters)) / d
    # print Rij
    except:
        Rij = 0
    return Rij


def compute_R(i, x, labels, clusters, nc):
    list_r = []
    for i in range(nc):
        for j in range(nc):
            if (i != j):
                temp = compute_Rij(i, j, x, labels, clusters, nc)
                list_r.append(temp)

    return max(list_r)


def compute_DB_index(x, labels, clusters, nc):
    # print x
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_R(i, x, labels, clusters, nc)

    DB_index = float(sigma_R) / float(nc)
    return DB_index


def main():
    df = pd.read_csv("dataset.csv")
    df = df.dropna()
    # print df
    x1 = df.copy()
    del x1['Customer']
    del x1['Effective To Date']
    x4 = pd.get_dummies(x1)
    # print x4
    n = 10
    clf = k_means(x4, n_clusters=n)
    centroids = clf[0]
    # 10 clusters
    labels = clf[1]
    # print x4[1]
    index_db_val = compute_DB_index(x4, labels, centroids, n)
    print("The value of Davies Bouldin index for a K-Means cluser of size " + str(n) + " is: " + str(index_db_val))


if __name__ == "__main__":
    main()
