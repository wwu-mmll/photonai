from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import MinimumPerformanceConstraint, DummyPerformanceConstraint, BestPerformanceConstraint, IntegerRange

import matplotlib.pyplot as plt

my_pipe = Hyperpipe(name='constrained_forest_pipe',
                    optimizer='grid_search',
                    metrics=['mean_squared_error', 'mean_absolute_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3, shuffle=True),
                    inner_cv=KFold(n_splits=10),
                    use_test_set=True,
                    verbosity=1,
                    project_folder='./tmp',
                    # output_settings=OutputSettings(mongodb_connect_url="mongodb://localhost:27017/photon_results",
                    #                               save_output=True),
                    performance_constraints=[DummyPerformanceConstraint('mean_absolute_error'),
                                             MinimumPerformanceConstraint('pearson_correlation', 0.65, 'any'),
                                             BestPerformanceConstraint('mean_squared_error', 3, 'mean')])


my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})

X, y = load_boston(return_X_y=True)
my_pipe.fit(X, y)


## plot Scatter plot

train_df = my_pipe.results_handler.get_mean_train_predictions()
pred_df = my_pipe.results_handler.get_test_predictions()

max_value = int(max(max(pred_df['y_true']), max(pred_df['y_pred']), max(train_df['y_pred'])))

fig, main_axes = plt.subplots()
main_axes.plot(range(max_value), range(max_value), color='black')
test_set = main_axes.scatter(pred_df["y_true"], pred_df["y_pred"], label="Test")
train_set = main_axes.scatter(train_df["y_true"], train_df["y_pred"], label="Training")
main_axes.legend(handles=[test_set, train_set], loc='lower right')
main_axes.set_xlabel("y true")
main_axes.set_ylabel("y predicted")

plt.show()
