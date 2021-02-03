from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import IntegerRange

X, y = load_boston(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe_no_performance',
                    optimizer='random_grid_search',
                    metrics=['mean_squared_error', 'pearson_correlation', 'mean_absolute_error', 'explained_variance'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA',
                           hyperparameters={'n_components': [0.5, 0.8, 0.3]})

# engage and optimize the good old SVM for Classification
my_pipe += PipelineElement('RandomForestRegressor',
                           hyperparameters={'n_estimators': IntegerRange(10, 50)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
