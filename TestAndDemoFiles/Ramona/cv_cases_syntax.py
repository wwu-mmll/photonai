# DO NOT RUN THIS FILE!!! JuST FOR SYNTAX DECLARATION

from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement, PipelineSwitch, PipelineFusion
from sklearn.model_selection import KFold, ShuffleSplit

X=[]
y=[]

my_pipe_manager = Hyperpipe('my_simple_pipe', metrics=['accuracy'],
                            optimizer='grid_search', optimizer_params={},
                            hyperparameter_specific_config_cv_object=KFold(n_splits=3),
                            hyperparameter_search_cv_object=ShuffleSplit(n_splits=2, test_size=0.2))

#CASE A: Simple Hyperparameter Optimization
my_pipe_manager.add(PipelineElement.create('pca', {'n_components': [10, 20, 30]}, whiten=False, test_disabled=True))
my_pipe_manager.add(PipelineElement.create('auto_encoder'), {'n_layers': [1, 2]})
my_pipe_manager.fit(X,y)


#CASE B:


