import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
#from skopt import Optimizer
#from skopt.optimizer import dummy_minimize
#from skopt import dummy_minimize
import scipy.io as sio
import keras
from photonai import Hyperpipe
from photonai import PipelineElement
from photonai import PhotonRegister
from photonai.base.PhotonBatchElement import PhotonBatchElement
from photonai.validation import ResultsTreeHandler
from photonai.neuro.BrainAtlas import AtlasLibrary
from scipy.stats import itemfreq
from photonai.investigator.Investigator import Investigator
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image
import time


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# RandomCtrlData = np.ones((1792, 121, 145, 121))
# RandomCtrlData = np.ones((172, 121, 145, 121))
# RandomCtrlLabels = np.random.randn((RandomCtrlData.shape[0]))

root_folder = '/spm-data/Scratch/spielwiese_ramona/PAC2018/'
filename = 'PAC2018_age.csv'
df = pd.read_csv(os.path.join(root_folder, filename))

X = df["PAC_ID"]
X = [os.path.join(root_folder, 'data_all/' + x + ".nii") for x in X]
y = df["Age"].values

X = X[0:150]
y = y[0:150]

#
# PhotonRegister.save(photon_name='Brain_Age_Splitting_Wrapper',
#                         class_str='photonai.modelwrapper.Brain_Age_Splitting_Wrapper.Brain_Age_Splitting_Wrapper', element_type="Transformer")
#
# PhotonRegister.save(photon_name='Brain_Age_Splitting_CNN',
#                         class_str='photonai.modelwrapper.Brain_Age_Splitting_CNN.Brain_Age_Splitting_CNN', element_type="Estimator")
#
# PhotonRegister.save(photon_name='Brain_Age_Random_Forest',
#                         class_str='photonai.modelwrapper.Brain_Age_Random_Forest.Brain_Age_Random_Forest', element_type="Estimator")

my_pipe = Hyperpipe('BrainAgePipe',
                        optimizer='grid_search',
                        metrics=['mean_absolute_error'],
                        best_config_metric='mean_absolute_error',
                        inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        outer_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        eval_final_performance=False)

# transformer = PipelineElement(, hyperparameters={})
# base_element=transformer
batched_transformer = PhotonBatchElement("PatchImages", hyperparameters={'patch_size': [25, 10]}, batch_size=10,
                                         nr_of_processes=10)
my_pipe += batched_transformer

#my_pipe += PipelineElement('Brain_Age_Splitting_Wrapper')

my_pipe += PipelineElement('Brain_Age_Random_Forest')

my_pipe.fit(X, y)




