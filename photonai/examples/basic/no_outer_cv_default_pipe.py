from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from sklearn.model_selection import KFold
import numpy as np
from sklearn.datasets import load_boston
X, y = load_boston(True)


os = OutputSettings(save_predictions='all', plots=False)

my_pipe = Hyperpipe(name='default_pipe',  # the name of your pipeline
                    metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],  # the performance metrics of interest
                    best_config_metric='mean_absolute_error',  # after hyperparameter search, this metric determined the winner config
                    eval_final_performance=False,
                    inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),  # test each configuration k times respectively
                    verbosity=1,
                    output_settings=os)


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe += PipelineElement('SimpleImputer', missing_values=np.nan, strategy='median')
my_pipe += PipelineElement('StandardScaler')


# my_pipe += PipelineElement('SVR', C=10, kernel='rbf')
# my_pipe += PipelineElement('LinearSVR', C=10, max_iter=100000)
my_pipe += PipelineElement('GaussianProcessRegressor')
# my_pipe += PipelineElement('RandomForestRegressor')
# my_pipe += PipelineElement('Ridge')


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# find mean and std of all metrics here
test_metrics = my_pipe.results.best_config.metrics_test
train_metrics = my_pipe.results.best_config.metrics_train
