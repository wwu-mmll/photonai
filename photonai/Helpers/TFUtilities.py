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


def one_hot_to_binary(one_hot_matrix):
    out = np.zeros((one_hot_matrix.shape[0]))
    for i in range(one_hot_matrix.shape[0]):
        out[i] = np.nonzero(one_hot_matrix[i,:])[0]
    return out

def binary_to_one_hot(binary_vector):
    classes = np.unique(binary_vector)
    out = np.zeros((binary_vector.shape[0],len(classes)),  dtype=np.int)
    for i, c in enumerate(classes):
        out[binary_vector==c,i] = 1
    return out