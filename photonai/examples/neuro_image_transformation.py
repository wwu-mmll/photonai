from ..base.PhotonBase import Hyperpipe, PipelineElement
from ..neuro.NeuroBase import NeuroBatch
from ..neuro.ImageBasics import ResampleImages, SmoothImages
from sklearn.model_selection import KFold



my_pipe = Hyperpipe('basic_svm_pipe_no_performance',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],  # the performance metrics of your interest
                    best_config_metric='accuracy',  # after hyperparameter search, the metric declares the winner config
                    outer_cv=KFold(n_splits=3),  # repeat hyperparameter search three times
                    inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively,
                    verbosity=1)


