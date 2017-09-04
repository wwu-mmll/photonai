import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from Framework.PhotonBase import PipelineElement, Hyperpipe
from Logging.Logger import Logger

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
random.seed(42)

pca_n_components = [10, 5]
svc_c = [1,10]
svc_kernel = "rbf"
# SET UP HYPERPIPE and choose accuracy as optimizer metric
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                    metrics=['accuracy', 'precision', 'f1_score'],
                    hyperparameter_specific_config_cv_object=KFold(n_splits=3),
                    hyperparameter_search_cv_object=KFold(n_splits=3),
                    eval_final_performance = True,
                    best_config_metric='accuracy', verbose=2)

my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('pca', {'n_components': pca_n_components})
my_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': [svc_kernel]})

# START HYPERPARAMETER SEARCH
Logger().info('-----------------')
Logger().info('OPTIMIZER METRIC: ACCURACY\n\n\n')
my_pipe.fit(X, y)

# SET UP HYPERPIPE and choose precision as optimizer metric
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                    metrics=['accuracy', 'precision', 'f1_score'],
                    hyperparameter_specific_config_cv_object=KFold(n_splits=3),
                    hyperparameter_search_cv_object=KFold(n_splits=3),
                    eval_final_performance = True,
                    best_config_metric='mean_squared_error')

my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('pca', {'n_components': pca_n_components})
my_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': [svc_kernel]})

# START HYPERPARAMETER SEARCH
Logger().info('\n\n\n-----------------')
Logger().info('OPTIMIZER METRIC: PRECISION\n\n\n')
my_pipe.fit(X, y)
Logger().info(my_pipe.test_performances)