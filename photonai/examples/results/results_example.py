from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from photonai.validation.ResultsTreeHandler import ResultsHandler
import matplotlib.pylab as plt


# WE USE THE BOSTON HOUSING DATA FROM SKLEARN
X, y = load_boston(True)


# DESIGN YOUR PIPELINE
settings = OutputSettings(save_feature_importances='best', save_predictions='best')

my_pipe = Hyperpipe('results_tree_example',
                    optimizer='sk_opt',  # which optimizer PHOTON shall use, in this case sk_opt
                    optimizer_params={'num_iterations': 20, 'acq_func_kwargs': {'kappa': 1}},
                    #optimizer='random_grid_search',  # which optimizer PHOTON shall use, in this case sk_opt
                    #optimizer_params={'k': 40},
                    metrics=['mean_squared_error'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)



# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# engage and optimize SVR
my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(1e-3, 100, range_type='geomspace', step=100),
                                                   'epsilon': FloatRange(1e-3, 10),
                                                   'tol': FloatRange(1e-4, 1e-2)}, kernel='linear')
# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

handler = ResultsHandler(my_pipe.results)

# get predictions for your best configuration (for all outer folds)
best_config_preds = handler.get_val_preds()
y_pred = best_config_preds['y_pred']
y_pred_probabilities = best_config_preds['y_pred_probabilities']
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
                               file='optimizer_history_scikit_optimize_20_scatter.png')

debug = True
