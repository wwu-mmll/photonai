from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import MinimumPerformanceConstraint, DummyPerformanceConstraint, BestPerformanceConstraint, IntegerRange


my_pipe = Hyperpipe(name='constrained_forest_pipe',
                    optimizer='grid_search',
                    metrics=['mean_squared_error', 'mean_absolute_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=10),
                    use_test_set=True,
                    verbosity=1,
                    project_folder='./tmp',
                    output_settings=OutputSettings(mongodb_connect_url="mongodb://localhost:27017/photon_results",
                                                   save_output=True),
                    performance_constraints=[DummyPerformanceConstraint('mean_absolute_error'),
                                             MinimumPerformanceConstraint('pearson_correlation', 0.65, 'any'),
                                             BestPerformanceConstraint('mean_squared_error', 3, 'mean')])


my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})

X, y = load_boston(return_X_y=True)
my_pipe.fit(X, y)
