from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from photonai.base.PhotonBatchElement import PhotonBatchElement
from sklearn.model_selection import KFold
import glob
import numpy as np


my_pipe = Hyperpipe('BatchAndSmooth',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error'],  # the performance metrics of your interest
                    best_config_metric='mean_absolute_error',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=2),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=2),  # test each configuration ten times respectively,
                    verbosity=1)


root_folder = "/spm-data/Scratch/spielwiese_ramona/test_photon_neuro/*.nii"
file_list = glob.glob(root_folder)
y = np.random.randint(145, size=len(file_list)).astype(np.float)

my_pipe += PhotonBatchElement("SmoothImages", hyperparameters={'fwhm': [[2, 2, 2], [3, 3, 3], [4, 4, 4]]},
                              batch_size=2)
my_pipe += PhotonBatchElement("ResampleImages", hyperparameters={'voxel_size': [[3, 3, 3], [2, 2, 2], [5, 5, 5]]},
                              batch_size=2, output_img=False)

my_pipe += PipelineElement('SVR')

my_pipe.fit(file_list[:10], y[:10])

debug = True
