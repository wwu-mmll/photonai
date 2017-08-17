"""
Utility functions for TensorFlow
Nils
"""
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matlab_util as mu
from skimage.measure import compare_ssim as SSIM
import matplotlib.pyplot as plt

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


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def scaleZeroOne(x, fixed_scale=True):
    if fixed_scale == False:
        if np.min(x) < 0:
            x = x + abs(np.min(x))
        else:
            x = x - np.min(x)
        x = x / np.max(x)
    else:
        x = x + 1
        x = x / 2
    return x


def corrThreshold(x, threshold=0.2):
    x[abs(x) < threshold] = 0
    return x



def applyPCA(X_train, X_test, n_components, rescale=False):
    print("Preprocessing data: PCA with", n_components, "components...")
    pca = PCA(whiten=True, n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test= pca.transform(X_test)
    if rescale == True:
        X_train = scaleZeroOne(X_train)
        X_test = scaleZeroOne(X_test)
    print("PCA explained variance sum:", np.sum(pca.explained_variance_ratio_))
    return X_train, X_test


def load_data_kirsten(data_path):
    from corr_timeseries import corr_timeseries
    import scipy.io as sio
    import matlab_util as mu
    import pandas as pd
    print('Loading labeled data from ...')
    print(data_path, '\n')

    # load intelligence scores
    df = pd.read_csv(data_path + '01_FSIQ_and_Pheno_Vals_N309.csv', delimiter=';',
                     header=0)
    df = df.values
    fsiq = df[:, 5]
    # load time series and correlate them
    input_ts = sio.loadmat(data_path + 'Extracted_MeanTS_of_Dosenbach_ROIs_Enh_NKI.mat')
    index_rois = sio.loadmat(data_path + 'kirsten_intel_rois_indices.mat')
    index_rois = np.squeeze(np.asarray(index_rois['roi_sort_indices'] - 1))
    input_ts = input_ts['ts_mean']
    # sort ROIs
    input_ts = input_ts[:, index_rois, :]
    # compute corr matrix
    conn_mat = corr_timeseries(input_ts)
    data_intel = np.empty([conn_mat.shape[0], 12720])
    for s in range(conn_mat.shape[0]):
        data_intel[s, :] = mu.mat2vec(conn_mat[1, :, :])

    return fsiq, data_intel


def prepareResultsFile(results, results_dir, results_filename, title_left=False):
    keys = list(results.keys())
    if title_left == True:
        string = ' ;'
    else:
        string = ''
    for i in range(len(keys)):
        string = string + keys[i] + ';'
    string = string + '\n'

    f = open((results_dir + "/" + results_filename), "a")
    f.write(string)
    f.close()


def writeResults(results, results_dir, results_filename, title_left=0):
    import csv
    if title_left != 0:
        results['title_left'] = title_left
    keys = list(results.keys())
    with open(results_dir + "/" + results_filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writerow(results)


def writeHeader(header, results_dir, results_filename):
    import csv
    empty = {'empty': ''}
    keys = list(empty.keys())
    with open(results_dir + "/" + results_filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writerow(empty)

    dict = {'header': header}
    keys = list(dict.keys())
    with open(results_dir + "/" + results_filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writerow(dict)


def measureSimilarity(real, reconstruction):
    print('Measure similarity...')
    results = {}
    # RMSE
    results['RMSE'] = RMSE(real, reconstruction)

    # SSIM
    ssim = []
    for i in range(real.shape[0]):
        real2d = mu.mat2vec(real[i, :], reverse=True)
        real2d[np.isnan(real2d)] = 0
        reconstruction2d = mu.mat2vec(reconstruction[i, :], reverse=True)
        reconstruction2d[np.isnan(reconstruction2d)] = 0
        ssim.append(SSIM(real2d, reconstruction2d))
    results['SSIM'] = np.mean(ssim)

    # MI

    # Graph Laplacian RMSE

    # Graph Laplacian SSIM

    # Graph Edit Distance

    # Global Efficiency

    return results


def RMSE(A,B):
    rmse = []
    for i in range(A.shape[0]):
        rmse.append(np.sqrt(np.mean(((A[i,:] - B[i,:]) ** 2))))
    return np.mean(rmse)


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def save_conn_plots(x, recon, loss=0, file='tmp/conn_plots.png', title=''):
    """Create a pyplot plot and save to buffer."""
    # figure = plt.figure()
    n_examples = 5
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))
    for example_i in range(n_examples):
        mat = mu.mat2vec(x[example_i, :], reverse=True)
        axs[0][example_i].imshow(mat, vmax=1, vmin=0.15, interpolation='none')
        mat2 = mu.mat2vec(recon[example_i, :], reverse=True)
        axs[1][example_i].imshow(mat2, vmax=1, vmin=0.15, interpolation='none')
        #
        # mat = x[example_i,:].reshape(28,28)
        # axs[0][example_i].imshow(mat)
        # mat2 = recon[example_i, :].reshape(28, 28)
        # axs[1][example_i].imshow(mat2)
    title = "Conn Matrices " + title + ": Orig vs Recon (Loss Epoch: {:.5f})".format(loss)
    plt.suptitle(title, fontsize=20)
    plt.savefig(file, format='png')
    plt.close()


def tf_mat2vec(input, reverse=True):
    if reverse == True:
        vecDim = input.get_shape().as_list()[0]
        matDim = .5  + np.sqrt(0.25 + 2 * vecDim)
        out = tf.placeholder(tf.float32, shape=[None, matDim, matDim], name='Input_2d')

        r = 0
        for i in range(int(matDim)):
            for j in range(int(matDim)):
                if j > i:
                    out[i, j] = input[r]
                    out[j, i] = input[r]
                    r += 1
        return out
    if reverse == False:
        pass


def multipleANOVA(X, group_labels, title_left=''):
    from scipy import stats
    from collections import OrderedDict
    print("Run ANOVAs for Site and Database Effects...")
    group_labels = group_labels.astype('int')
    F = np.empty(X.shape[1])
    p = np.empty(X.shape[1])
    groups = {}
    string = ''
    results = OrderedDict({})

    for k, g in enumerate(np.unique(group_labels)):
        if not k == 0:
            groups[str(g)] = X[group_labels == g, :]
            string = string + 'groups["' + str(g) + '"][:,i],'
    for i in range(X.shape[1]):
        exec('F[i], p[i] = stats.f_oneway(' + string + ')')
    results['condition'] = title_left
    results['n_F_sig'] = np.sum(p<0.05)
    results['F_max'] = np.max(F)
    results['F_mean'] = np.mean(F)
    results['F_std'] = np.std(F)

    return results

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