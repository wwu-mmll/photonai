"""
Utility functions for TensorFlow
Nils
"""
import numpy as np

def get_batches(x, batch_size, shuffle=True):
    batches = []
    n_data = x.shape[0]
    n_batches = int(np.ceil(n_data/batch_size))

    if shuffle == 1:
        indices = np.random.permutation(n_data)
    else:
        indices = np.arange(0, n_data)

    for ind_batch in range(n_batches-1):
        batches.append(indices[ind_batch*batch_size:ind_batch*batch_size+batch_size])
    batches.append(indices[batch_size*(n_batches-1):,])

    return batches, n_batches


def oneHot(y, reverse=False):
    if reverse == False:
        classes = np.unique(y)
        out = np.zeros((y.shape[0],len(classes)),  dtype=np.int)
        for i, c in enumerate(classes):
            out[y==c,i] = 1
    else:
        out = np.zeros((y.shape[0]))
        for i in range(y.shape[0]):
            out[i] = np.nonzero(y[i,:])[0]
    return out