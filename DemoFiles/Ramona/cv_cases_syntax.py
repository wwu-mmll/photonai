# DO NOT RUN THIS FILE!!! JuST FOR SYNTAX DECLARATION

from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch, PipelineStacking
from sklearn.model_selection import KFold, ShuffleSplit

X=[]
y=[]

my_pipe_manager = Hyperpipe('my_simple_pipe', metrics=['accuracy'],
                            optimizer='grid_search', optimizer_params={},
                            inner_cv=KFold(n_splits=3),
                            outer_cv=ShuffleSplit(n_splits=2, test_size=0.2))

#CASE A: Simple Hyperparameter Optimization
my_pipe_manager.add(PipelineElement.create('pca', {'n_components': [10, 20, 30]}, whiten=False, test_disabled=True))
my_pipe_manager.add(PipelineElement.create('auto_encoder'), {'n_layers': [1, 2]})
my_pipe_manager.fit(X,y)


#CASE B:


