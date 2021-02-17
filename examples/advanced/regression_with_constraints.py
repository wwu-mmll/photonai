from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import MinimumPerformanceConstraint, IntegerRange

X, y = load_boston(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe(name='basic_svm_pipe_no_performance',
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 25},
                    metrics=['mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=3),
                    use_test_set=True,
                    verbosity=1,
                    project_folder='./tmp',
                    output_settings=OutputSettings(mongodb_connect_url="mongodb://localhost:27017/photon_results",
                                                   save_output=True),
                    performance_constraints=[MinimumPerformanceConstraint('mean_squared_error', 35, 'first'),
                                             MinimumPerformanceConstraint('pearson_correlation', 0.7, 'any')])


my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})


# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
