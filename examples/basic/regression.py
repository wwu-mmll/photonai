from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import IntegerRange, FloatRange


my_pipe = Hyperpipe('basic_regression_pipe',
                    optimizer='random_search',
                    optimizer_params={'n_configurations': 25},
                    metrics=['mean_squared_error', 'mean_absolute_error', 'explained_variance'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=3, shuffle=True),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe += PipelineElement('SimpleImputer')
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('LassoFeatureSelection',
                           hyperparameters={'percentile': [0.1, 0.2, 0.3],
                                            'alpha': FloatRange(0.5, 5)})

my_pipe += PipelineElement('RandomForestRegressor',
                           hyperparameters={'n_estimators': IntegerRange(10, 50)})

# load data and train
X, y = load_boston(return_X_y=True)
my_pipe.fit(X, y)
