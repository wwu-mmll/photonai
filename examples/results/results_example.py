from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange

# WE USE THE BOSTON HOUSING DATA FROM SKLEARN
X, y = load_boston(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('results_example',
                    learning_curves=True,
                    learning_curves_cut=FloatRange(0, 1, 'range', 0.2),
                    optimizer='sk_opt',
                    optimizer_params={'n_configurations': 20, 'acq_func_kwargs': {'kappa': 1}},
                    metrics=['mean_squared_error'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp')

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# engage and optimize SVR
my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(1e-3, 100, range_type='logspace'),
                                                   'epsilon': FloatRange(1e-3, 10),
                                                   'tol': FloatRange(1e-4, 1e-2)}, kernel='linear')
# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

handler = my_pipe.results_handler

# get predictions for your best configuration (for all outer folds)
best_config_preds = handler.get_test_predictions()
y_pred = best_config_preds['y_pred']
y_pred_probabilities = best_config_preds['probabilities']
y_true = best_config_preds['y_true']

# get feature importances (training set) for your best configuration (for all outer folds)
# this function returns the importance scores for the best configuration of each outer fold in a list
importance_scores = handler.get_importance_scores()

# get performance for all outer folds
performance = handler.get_performance_outer_folds()

# get all configuration evaluations
config_evaluations = handler.get_config_evaluations()
minimum_config_evaluations = handler.get_minimum_config_evaluations()

# handler.plot_optimizer_history('mean_squared_error', 'RGS 40 Eval (Scatter)', 'scatter',
#                                'optimizer_history_random_grid_search_40_scatter.png')
handler.plot_optimizer_history(metric='mean_squared_error',
                               title='Scikit Optimize 20 Eval (Scatter)',
                               type='scatter',
                               reduce_scatter_by=1,
                               file='./tmp/optimizer_history_scikit_optimize_20_scatter.png')

# plot learning curves of all configs of the first outer fold
handler.plot_learning_curves_outer_fold(outer_fold_nr=1, config_nr_list=None, save=False, show=True)

debug = True
