from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PipelineBranch, PipelineStack, PipelineSwitch, SourceFilter
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from sklearn.model_selection import ShuffleSplit, KFold
import numpy as np
from sklearn.datasets import load_boston

# GET DATA: Boston Housing REGRESSION example
X, y = load_boston(True)

# DEFINE OUTPUT FOLDER
output_folder = './my_output_folder/'

# DESIGN YOUR PIPELINE
pipe = Hyperpipe(name='my_regression_example_pipe_basic_switch',
                 optimizer='grid_search',
                 metrics=['mean_absolute_error', 'mean_squared_error', 'spearman_correlation'],
                 best_config_metric='mean_absolute_error',
                 outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                 inner_cv=KFold(n_splits=3, shuffle=True),
                 eval_final_performance=True,
                 output_settings=OutputSettings(project_folder=output_folder),
                 verbosity=0)

pipe += PipelineElement('StandardScaler')

# setup pipeline branches with half of the features each fed into a PCA and an estimator
# if both PCAs are disabled, features are simply concatenated and passed to the final estimator
source1_branch = PipelineBranch('source1_features')
# first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
source1_branch += SourceFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1]/2))))
source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
source1_branch += PipelineElement('RandomForestRegressor')    # final estimator with branch output as features

source2_branch = PipelineBranch('source2_features')
# second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
source2_branch += SourceFilter(indices=np.arange(start=int(np.floor(X.shape[1]/2)), stop=X.shape[1]))
source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
source2_branch += PipelineElement('RandomForestRegressor')    # final estimator with branch output as features

# pick the better branch
pipe += PipelineSwitch('my_branch_switch', pipeline_element_list=[source1_branch, source2_branch])

pipe.fit(X, y)
