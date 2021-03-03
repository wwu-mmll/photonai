from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange

X, y = load_boston(return_X_y=True)

my_pipe = Hyperpipe('feature_selection',
                    optimizer='grid_search',
                    metrics=['mean_squared_error', 'pearson_correlation', 'mean_absolute_error', 'explained_variance'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')

lasso = PipelineElement('LassoFeatureSelection',
                        hyperparameters={'percentile': [0.1, 0.2, 0.3]}, alpha=1)
f_regression = PipelineElement('FRegressionSelectPercentile',
                               hyperparameters={'percentile': [10, 20, 30]})

my_pipe += Switch('FeatureSelection', [lasso, f_regression])
my_pipe += PipelineElement('RandomForestRegressor',
                           hyperparameters={'n_estimators': IntegerRange(10, 50)})
my_pipe.fit(X, y)
