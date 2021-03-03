import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement

X, y = load_boston(return_X_y=True)

my_pipe = Hyperpipe(name='single_outer_pipe',
                    metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_absolute_error',
                    use_test_set=False,
                    inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                    verbosity=0,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('SimpleImputer', missing_values=np.nan, strategy='median')
my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('GaussianProcessRegressor')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# find mean and std of all metrics here
test_metrics = my_pipe.results.best_config.metrics_test
train_metrics = my_pipe.results.best_config.metrics_train
