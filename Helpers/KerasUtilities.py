from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten, AlphaDropout
from keras.layers.advanced_activations import PReLU
from keras.layers import Conv3D, MaxPooling3D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
import keras as keras 
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd 

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, confusion_matrix
from sklearn.cross_validation import KFold


# merge _reg and _classif at some point
def create_model_reg(input_size, layer_sizes=[], actFunc='relu', learning_rate=0.001, dropout_rate=0, 
                     batch_normalization=1, nb_epoch=200, loss_function='mean_absolute_error', 
                     metrics=['mean_absolute_error'], optimizer='adam', gpu_device='/gpu:0', kernel_initializer='lecun_normal'):
    # create model
    # print('Dropout:', dropout_rate, 'PCA components:', input_size, 'learning_rate:', learning_rate)
    # print('Constructing KERAS model with', '\n', 'input_size', input_size, '\n', 'activationFunction', actFunc, '\n', 'layer_sizes', layer_sizes, '\n', 'learning_rate', learning_rate, '\n', 'dropout_rate', dropout_rate, '\n','batch_normalization', batch_normalization, '\n', 'nb_epoch', nb_epoch)
    if actFunc == 'selu':
        DropItLikeItsHot = AlphaDropout
    else:
        DropItLikeItsHot = Dropout
        
    model = Sequential()
    input_dim = input_size
    for i, dim in enumerate(layer_sizes):
        with tf.device(gpu_device):
            if i == 0:
                model.add(Dense(dim, input_dim=input_dim, kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(dim, kernel_initializer=kernel_initializer))

        if batch_normalization == 1:
            model.add(BatchNormalization())
        with tf.device(gpu_device):
            if actFunc == 'prelu':
                from keras.layers.advanced_activations import PReLU
                model.add(PReLU(init='zero', weights=None))
            else:
                model.add(Activation(actFunc))

            if dropout_rate > 0:
                model.add(DropItLikeItsHot(dropout_rate))
    with tf.device(gpu_device):
        model.add(Dense(1, activation='linear'))
        
    # Compile model
    optimizer = define_optimizer(optimizer_type=optimizer, lr=learning_rate)
    
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model

def create_model_classif(input_size, n_classes, layer_sizes=[], actFunc = 'relu', learning_rate=0.001, 
                         dropout_rate=0, batch_normalization=False, do_class_weights=False, nb_epoch=200, class_weights={0:1,1:1},
                         loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam',gpu_device='/gpu:0'):
    # create model
    #print('Dropout:', dropout_rate, 'PCA components:', input_size, 'learning_rate:', learning_rate)
    #print('Constructing KERAS model with', '\n', 'input_size', input_size, '\n', 'activationFunction', actFunc, '\n', 'layer_sizes', layer_sizes, '\n', 'learning_rate', learning_rate, '\n', 'dropout_rate', dropout_rate, '\n','batch_normalization', batch_normalization, '\n', 'do_class_weights', do_class_weights,'\n', 'nb_epoch', nb_epoch)
    model = Sequential()
    input_dim = input_size
    for i, dim in enumerate(layer_sizes):
        with tf.device(gpu_device):
            if i == 0:
                model.add(Dense(dim, input_dim=input_dim, kernel_initializer='uniform'))
            else:
                model.add(Dense(dim, kernel_initializer='uniform'))
        
        if batch_normalization == True:
            model.add(BatchNormalization())
        with tf.device(gpu_device):
            if actFunc == 'prelu':
                from keras.layers.advanced_activations import PReLU
                model.add(PReLU(init='zero', weights=None))
            else:
                model.add(Activation(actFunc))
        
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate)) 
    
    with tf.device(gpu_device):
        model.add(Dense(n_classes, activation='softmax'))
    
    # Compile model
    optimizer = define_optimizer(optimizer_type=optimizer, lr=learning_rate)
    
    #if do_class_weights == True:
    #    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)#, class_weights=class_weights)
    #else:
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    ######### OLD VERSION! DO NOT USE ##########
    #if do_class_weights == 1:
    #    model.compile(loss='categorical_crossentropy', optimizer='adam', lr=learning_rate, metrics=['accuracy'], 
    #                  class_weights=class_weights)
    #else:
    #    model.compile(loss='categorical_crossentropy', optimizer='adam', lr=learning_rate, metrics=['accuracy'])
    #print(model.summary())
    return model

def create_snn_classif(input_size, n_classes, layer_sizes=[], actFunc = 'selu', learning_rate=0.001, 
                         alpha_dropout_rate=0, nb_epoch=200, 
                         loss='categorical_crossentropy', metrics=['accuracy'],
                         optimizer='adam',gpu_device='/gpu:0'):

    model = Sequential()
    input_dim = input_size
    for i, dim in enumerate(layer_sizes):
        with tf.device(gpu_device):
            if i == 0:
                model.add(Dense(dim, input_dim=input_dim, kernel_initializer='lecun_normal'))
            else:
                model.add(Dense(dim, kernel_initializer='lecun_normal'))

        with tf.device(gpu_device):
            model.add(Activation(actFunc))
        
            if alpha_dropout_rate > 0:
                model.add(AlphaDropout(alpha_dropout_rate)) 
    
    with tf.device(gpu_device):
        model.add(Dense(n_classes, activation='softmax'))
    
    # Compile model
    optimizer = define_optimizer(optimizer_type=optimizer, lr=learning_rate)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def create_dnn(X, y, layer_sizes:list, classification:bool, loss=None, act_func='selu', learning_rate=0.001, dropout_rate=0,
               metrics=None, optimizer='adam', kernel_initializer='lecun_normal', batch_normalization=True,
               gpu_device='/gpu:0'):
    
    # check for some stuff
    if act_func == 'selu':
        DropItLikeItsHot = AlphaDropout
    else:
        DropItLikeItsHot = Dropout
    
    if classification:
        last_activation = 'softmax'
        try:
            n_neurons_last_layer = y.shape[1]
        except:
            n_neurons_last_layer = len(np.unique(y))
        if not loss:
            loss = 'categorical_crossentropy'
    else:
        last_activation = 'linear'
        n_neurons_last_layer = 1
        if not loss:
            loss = 'mean_squared_error'
    
    model = Sequential()
    
    for i, dim in enumerate(layer_sizes):
        with tf.device(gpu_device):
            if i == 0:
                model.add(Dense(dim, input_dim=X.shape[1], kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(dim, kernel_initializer=kernel_initializer))

        if batch_normalization:
            model.add(BatchNormalization())
        
        with tf.device(gpu_device):
            if act_func == 'prelu':
                model.add(PReLU(init='zero', weights=None))
            else:
                model.add(Activation(act_func))
                
            if i+1 < len(layer_sizes):
                if dropout_rate > 0:
                    model.add(DropItLikeItsHot(dropout_rate))
    
    with tf.device(gpu_device):
        model.add(Dense(n_neurons_last_layer, activation=last_activation))
        
    # Compile model
    optimizer = define_optimizer(optimizer_type=optimizer, lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    
    return model



def create_model_3dcnn(input_shape=[], n_classes=[], n_filters=[], kernel_sizes=[], n_convolutions_per_block=[], pooling_size=[],
                       strides=[], size_last_layer=[], actFunc='relu', learning_rate=0.001, dropout_rate=0, batch_normalization=False, 
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
    if batch_normalization:
        model.add(BatchNormalization())
    with tf.device(gpu_device):
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    optimizer = define_optimizer(optimizer_type=optimizer, lr=learning_rate)

    # Let's train the model using RMSprop
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model
    

def define_optimizer(optimizer_type='adam',lr=0.001):
    # TO DO: - use *kwargs to allow specification of additional optimizer hyperparameters
    #        - learn optimizer, see DeepMind paper
    #        - enable tf custom optimizer use
    if optimizer_type.lower() == 'adam':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    elif optimizer_type.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    
    elif optimizer_type.lower() == 'rmsprop':   
        optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
        
    elif optimizer_type.lower() == 'adagrad':
        optimizer = keras.optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=0.0)
    
    elif optimizer_type.lower() == 'adadelta':
        optimizer = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)

    elif optimizer_type.lower() == 'adamax':
        optimizer = keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    elif optimizer_type.lower() == 'nadam':
        optimizer = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    
    else:
        raise ValueError('Choose a valid keras optimizer!')
    
    return optimizer 
        
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    np.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def dumbest_model_classif(targets, train, val, test):
    # TO DO: only get ratio for training set
    #        get biggest group, put all subjects in this group, independent of cv set
    print('GET INFORMATION ON DUMBEST CLASSIFICATION\n')
    
    # Catch if targets are not one hot encoded
    if len(targets.shape) == 1:
        import tf_utils as tfu
        targets = tfu.oneHot(targets)
    
    #loop over sets
    sets = [train, val, test]
    set_names = ['train', 'validation', 'test']
    cnt = 0
    for cv_set in sets:
        # class_sizes is an empty list
        # loop over all classes and get number of subjects in that class
        # add it to list
        class_sizes = []
        targets_set = targets[cv_set,:]
        set_size = targets_set.shape[0]
        for i in range(targets_set.shape[1]):
            class_sizes.append(np.sum(targets_set[:,i]==1))
        ratio = np.amax(class_sizes)/set_size
        print('CV Set: ', set_names[cnt])
        print('Ratio: ', ratio)
        cnt += 1
    
    
def dumbest_model_reg(targets, train, val, test):

    print('GET INFORMATION ON DUMBEST MODEL\n')

    yTrue = targets[val]
    yTrain = targets[train]
    trainMean = np.tile(np.mean(yTrain), (len(yTrue), 1))
    MAE_dumb = mean_absolute_error(yTrue, trainMean)
    MAE_dumb2 = median_absolute_error(yTrue, trainMean)
    MSE_dumb = mean_squared_error(yTrue, trainMean)
    print('Train Mean: ', trainMean[0], '\n')
    print('Mean Absolute Error (val):', MAE_dumb)
    print('Median Absolute Error (val):', MAE_dumb2)
    print('Mean Squared Error (val):', MSE_dumb, '\n')

    yTrue = targets[test]
    trainMean = np.tile(np.mean(yTrain), (len(yTrue), 1))
    MAE_dumb = mean_absolute_error(yTrue, trainMean)
    MAE_dumb2 = median_absolute_error(yTrue, trainMean)
    MSE_dumb = mean_squared_error(yTrue, trainMean)
    print('Mean Absolute Error (test):', MAE_dumb)
    print('Median Absolute Error (test):', MAE_dumb2)
    print('Mean Squared Error (test):', MSE_dumb)

def print_hyperparams(hyperparams_dict, *hyperparams):
    names = list(hyperparams_dict.keys())
    string = 'HYPERPARAMETERS: '
    for i, value in enumerate(hyperparams):
        string = string + str(names[i]) + ': ' + str(value) + ';'

    print(string)
    

import nibabel as nib
class NiiLoader(object):
    
    def __call__(self, filepaths, vectorize=False, **kwargs):
        # loading all .nii-files in one folder is no longer supported

        if isinstance(filepaths, str):
            raise TypeError('Filepaths must be passed as list.')

        elif isinstance(filepaths, list):
            # iterate over and load every .nii file
            # this requires one nifti per subject
            img_data = []
            for ind_sub in range(len(filepaths)):
                img = nib.load(filepaths[ind_sub], mmap=False)
                img_data.append(img.get_data())

        else:
            raise TypeError('Filepaths must be passed as list.')


        # stack list elements to matrix
        data = np.stack(img_data, axis=0)
        if vectorize:
            data = np.reshape(data, (data.shape[0], data.shape[1] *
                                 data.shape[2] * data.shape[3]))
        return data

def get_rois(atlas='aal', rois=[]):
    if atlas.lower() == 'aal':
        # need to use image file
        # need to check voxel size, orientation... and reshape, re-orient masks if necessary
        
        aal = np.load('/home/nils/data/aal-79-95-69.npy')
        aal_names = pd.read_table('/home/nils/data/aal.txt', header=None)
        for i in range(aal_names.shape[0]):
            aal_names.iloc[i] = aal_names.values[i][0].replace(" ", "_")
        
        masks = {}
        if rois:
            # check if all roi_ids are positive
            if np.min(rois) < 1:
                raise ValueError('ROI IDs must not be <1!')
            for roi_id in rois:
                roi = np.zeros(aal.shape)
                roi[aal==roi_id] = 1
                masks[aal_names.values[roi_id-1][0]]  = roi
            
            roi = np.zeros(aal.shape)
            roi[aal!=0] = 1
            masks['whole_brain'] = roi
            
        else:   # if no roi_ids are supplied, use all rois
            all_rois = list(range(1,aal_names.shape[0]+1))
            masks = get_rois(atlas=atlas, rois = all_rois )
    else:
        raise ValueError('Currently AAL is the only valid Atlas!')
        masks = []
        
    return masks  

def crop_box_around_roi(data):
    true_points = np.argwhere(data[0])
    corner1 = true_points.min(axis=0)
    corner2 = true_points.max(axis=0)
    out = []
    for i in range(data.shape[0]):
        out.append(data[i, corner1[0]:corner2[0]+1,
                       corner1[1]:corner2[1]+1,
                       corner1[2]:corner2[2]+1])
    out = np.asarray(out)
    return out
