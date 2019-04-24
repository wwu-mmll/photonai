"""
===========================================================
Project: 
===========================================================
Description
-----------

Version
-------
Created:        DD-MM-YYYY
Last updated:   DD-MM-YYYY


Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.investigator.Investigator import Investigator
from photonai.configuration.Register import PhotonRegister
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
import pandas as pd
import os
import cv2


# GET DATA
def load_data(path, data_path):
    X = []
    y = []
    data_set = pd.read_csv(path)
    print('Read train images')
    for index, row in data_set.iterrows():
        image_path = os.path.join(data_path, str(row['img']))
        print(image_path)
        img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), (256, 192) ).astype(np.float32)/255
        X.append(img)
        y.append( [ row['rating'] ] )
        print ("Loading"+row['img'], end="\r")
    return np.array(X), np.array(y)


X_train, y_train = load_data('/home/nwinter/Downloads/Tim/Train6.csv', '/home/nwinter/Downloads/Tim')
#X_test, y_test = load_data('/home/nwinter/Downloads/Tim/Test7.csv', '/home/nwinter/Downloads/Tim')



# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('pretrained_cnn',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_squared_error'],  # the performance metrics of your interest
                    best_config_metric='mean_squared_error',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),  # repeat hyperparameter search three times
                    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2),  # test each configuration ten times respectively
                    verbosity=1) # get error, warn and info message                    )



# ADD ELEMENTS TO YOUR PIPELINE

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('PretrainedCNNRegressor', {}, input_shape=(192,256,3), freezing_point=8)


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X_train, y_train)