from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PipelineStack, PipelineSwitch
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.datasets import load_boston

# GET DATA: Boston Housing REGRESSION example
X, y = load_boston(True)

# DEFINE OUTPUT FOLDER
output_folder = './my_output_folder/'

# DESIGN YOUR PIPELINE
pipe = Hyperpipe(name='my_regression_example_pipe_multi_switch',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error', 'mean_squared_error', 'spearman_correlation'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 inner_cv=KFold(n_splits=3, shuffle=True),
                 eval_final_performance=True,
                 output_settings=OutputSettings(project_folder=output_folder),
                 verbosity=1)

pipe += PipelineElement('StandardScaler')

# setup switch to choose between PCA or simple feature selection and add it to the pipe
pre_switch = PipelineSwitch('preproc_switch')
pre_switch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
pre_switch += PipelineElement('FRegressionSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
pipe += pre_switch

# setup estimator switch and add it to the pipe
estimator_switch = PipelineSwitch('estimator_switch')
estimator_switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
estimator_switch += PipelineElement('RandomForestRegressor', hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
pipe += estimator_switch

#pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
pipe.fit(X, y)
