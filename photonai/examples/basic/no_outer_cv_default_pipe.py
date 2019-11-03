import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings

X, y = load_boston(True)

my_pipe = Hyperpipe(name='default_pipe',
                    metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],  # the performance metrics of interest
                    best_config_metric='mean_absolute_error',
                    eval_final_performance=False,
                    inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                    verbosity=2,
                    output_settings=OutputSettings(plots=False, project_folder='./tmp/'))


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('SimpleImputer', missing_values=np.nan, strategy='median')
my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('GaussianProcessRegressor')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# find mean and std of all metrics here
test_metrics = my_pipe.results.best_config.metrics_test
train_metrics = my_pipe.results.best_config.metrics_train
