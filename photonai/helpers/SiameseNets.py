import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def dot_prod(vects):
    x, y = vects
    return K.dot(x, y)


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(
                      K.maximum(margin - y_pred, 0)))


# ensures every subject is present the same number of times
def create_pairs(x, digit_indices, n_pairs_per_subject):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    # x: data, digit_indices: lists of indices of subjects in class1 and class2, respectively

    n_sample_pairs = 2 * n_pairs_per_subject * len(digit_indices[0]) * 2

    if n_sample_pairs > 100000:
        print('INSANE: You are trying to use ', n_sample_pairs,
              'sample pairs.')

    print('Generating ', n_sample_pairs, 'sample pairs.')

    pairs = []
    labels = []
    if not len(digit_indices[0]) == len(digit_indices[1]):
        raise ValueError('Lists do not have the same length.')

    pos_pairs = draw_pos_pairs(digit_indices, n_pairs_per_subject)
    neg_pairs = draw_neg_pairs(digit_indices, n_pairs_per_subject)

    for d in range(len(pos_pairs)):
        z1, z2 = pos_pairs[d]
        pairs += [[x[z1], x[z2]]]
        z1, z2 = neg_pairs[d]
        pairs += [[x[z1], x[z2]]]
        labels += [1, 0]
    return np.array(pairs), np.array(labels)


def draw_pos_pairs(indices, n_pairs_per_subject):
    pairs = []
    for ind_lists in range(2):
        a = indices[ind_lists]
        for ind_pair in range(n_pairs_per_subject):
            for ind_sub in range(len(a)):
                p1 = a[ind_sub]
                next_ind = (ind_sub + 1 + ind_pair) % len(a)
                p2 = a[next_ind]
                pairs.append([p1, p2])
    return pairs


def draw_neg_pairs(indices, n_pairs_per_subject):
    pairs = []
    a = indices[0]
    b = indices[1]
    for ind_lists in range(2):
        for ind_pair in range(n_pairs_per_subject):
            for ind_sub in range(len(a)):
                p1 = a[ind_sub]
                next_ind = (ind_sub + ind_pair + (ind_lists * (
                n_pairs_per_subject - 1)) + ind_lists) % len(b)
                p2 = b[next_ind]
                pairs.append([p1, p2])
    return pairs


def create_base_network(input_dim, actFunc):
    """base network to be shared (eq. to feature extraction).
    """
    seq = Sequential()
    seq.add(Dense(10, input_shape=(input_dim,), activation=actFunc))
    # seq.add(Dropout(0.6))
    #seq.add(Dense(5, activation=actFunc))
    return seq


def compute_accuracy(predictions, labels):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

def acc_t(y_true, y_pred):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(y_pred < 0.5), y_true)))

def create_base_3dcnn(input_shape=[], n_filters=[], kernel_sizes=[], n_convolutions_per_block=[], pooling_size=[],size_last_layer=[],
                       strides=[], actFunc='relu', learning_rate=0.001, dropout_rate=0, batch_normalization=False, 
                       nb_epoch=200, loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam',gpu_device='/gpu:0'):
    
   
    # size_last_layer
    # n_filters
    # kernel_size
    # strides
    # n_convolutions_per_block
    
    ks = kernel_sizes
    ps = pooling_size
    nf = n_filters
    input_shape = (input_shape[0],input_shape[1],input_shape[2],1)
    
    model = Sequential()

    for ind_blocks in range(len(n_filters)):
        for ind_convs in range(n_convolutions_per_block):
            if ind_blocks == 0 and ind_convs == 0:
                with tf.device(gpu_device):
                    model.add(Conv3D(nf[ind_blocks], [ks[0],ks[1],ks[2]], strides=strides, padding='same',
                                     input_shape=input_shape))
                    model.add(Activation(actFunc))
                if batch_normalization:
                    model.add(BatchNormalization())
            else:    
                with tf.device(gpu_device):
                    model.add(Conv3D(nf[ind_blocks], [ks[0],ks[1],ks[2]], strides=strides, padding='same'))
                    model.add(Activation(actFunc))

                if batch_normalization:
                    model.add(BatchNormalization())
        with tf.device(gpu_device):
            if ps:
                model.add(MaxPooling3D(pool_size=(ps, ps, ps)))

            if dropout_rate:
                model.add(Dropout(dropout_rate))

    with tf.device(gpu_device):
        model.add(Flatten())
        model.add(Dense(size_last_layer))
        model.add(Activation(actFunc))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
    if batch_normalization:
        model.add(BatchNormalization())


    print(model.summary())
    return model

def create_base_dnn_classif(input_size, layer_sizes=[], actFunc = 'relu', learning_rate=0.001, 
                         dropout_rate=0, batch_normalization=False, do_class_weights=False, nb_epoch=200, class_weights={0:1,1:1},
                         loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam',gpu_device='/gpu:0',l1=0,l2=0, 
                           weight_initializer='glorot_uniform'):
    # create model
    #print('Dropout:', dropout_rate, 'PCA components:', input_size, 'learning_rate:', learning_rate)
    #print('Constructing KERAS model with', '\n', 'input_size', input_size, '\n', 'activationFunction', actFunc, '\n', 'layer_sizes', layer_sizes, '\n', 'learning_rate', learning_rate, '\n', 'dropout_rate', dropout_rate, '\n','batch_normalization', batch_normalization, '\n', 'do_class_weights', do_class_weights,'\n', 'nb_epoch', nb_epoch)
    model = Sequential()
    input_dim = input_size
    for i, dim in enumerate(layer_sizes):
        with tf.device(gpu_device):
            if i == 0:
                model.add(Dense(dim, input_dim=input_dim, kernel_initializer=weight_initializer,kernel_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l1(l1)))
            else:
                model.add(Dense(dim, kernel_initializer=weight_initializer,kernel_regularizer=regularizers.l2(l2),
                activity_regularizer=regularizers.l1(l1)))
        
        if batch_normalization:
            model.add(BatchNormalization())
        with tf.device(gpu_device):
            if actFunc == 'prelu':
                from keras.layers.advanced_activations import PReLU
                model.add(PReLU(init='zero', weights=None))
            else:
                model.add(Activation(actFunc))
        
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate)) 

    print(model.summary())
    return model
