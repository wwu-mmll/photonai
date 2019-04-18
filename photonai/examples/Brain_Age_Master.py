import numpy as np
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


RandomCtrlData = np.ones((1792, 121, 145, 121))
RandomCtrlLabels = np.random.randn((RandomCtrlData.shape[0]))


#X = AtlasLibrary().get_nii_files_from_folder('/spm-data/Scratch/spielwiese_ramona/PAC2018/data/', extension='.nii')
#print(len(X))
#Brain_Images = image.load_img(X).get_data()
#Brain_Images = np.moveaxis(Brain_Images, 3, 0)
#print(Brain_Images.shape)



#label_dataframe = pd.read_excel(r'/spm-data/Scratch/spielwiese_ramona/PAC2018/PAC2018_Covariates_ALL.xlsx')
#age_labels = label_dataframe["Age"]




PhotonRegister.save(photon_name='Brain_Age_Splitting_Wrapper',
                        class_str='photonai.modelwrapper.Brain_Age_Splitting_Wrapper.Brain_Age_Splitting_Wrapper', element_type="Transformer")

PhotonRegister.save(photon_name='Brain_Age_Splitting_CNN',
                        class_str='photonai.modelwrapper.Brain_Age_Splitting_CNN.Brain_Age_Splitting_CNN', element_type="Estimator")

PhotonRegister.save(photon_name='Brain_Age_Random_Forest',
                        class_str='photonai.modelwrapper.Brain_Age_Random_Forest.Brain_Age_Random_Forest', element_type="Estimator")

my_pipe = Hyperpipe('BrainAgePipe',
                        optimizer='grid_search',
                        metrics=['accuracy'],
                        best_config_metric='accuracy',
                        inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        outer_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        eval_final_performance=False)

transformer = PipelineElement("Brain_Age_Splitting_Wrapper", hyperparameters={})
batched_transformer = PhotonBatchElement("batched_trans", batch_size=10, base_element=transformer)
my_pipe += batched_transformer

#my_pipe += PipelineElement('Brain_Age_Splitting_Wrapper')

my_pipe += PipelineElement('Brain_Age_Random_Forest')

my_pipe.fit(RandomCtrlData, RandomCtrlLabels)




