from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, ShuffleSplit
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler


# WE USE THE BOSTON HOUSING DATA FROM SKLEARN
X, y = load_boston(True)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('skopt_example',
                    optimizer='sk_opt',  # which optimizer PHOTON shall use, in this case sk_opt
                    optimizer_params={'num_iterations': 40, 'acq_func_kwargs': {'kappa': 1}},
                    #optimizer='random_grid_search',  # which optimizer PHOTON shall use, in this case sk_opt
                    #optimizer_params={'k': 40},
                    metrics=['mean_squared_error'],
                    best_config_metric='mean_squared_error',
                    outer_cv=ShuffleSplit(n_splits=5, test_size=0.2),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1)



# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# engage and optimize SVR
# linspace and logspace is converted to uniform and log-uniform priors in skopt
# my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(1e-3, 100, range_type='logspace'),
#                                                    'epsilon': FloatRange(1e-3, 10, range_type='logspace'),
#                                                    'tol': FloatRange(1e-4, 1e-2, range_type='linspace'),
#                                                    'kernel': Categorical(['linear', 'rbf', 'poly'])})
my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(1e-3, 100),
                                                   'epsilon': FloatRange(1e-3, 10),
                                                   'tol': FloatRange(1e-4, 1e-2),
                                                   'kernel': Categorical(['linear', 'rbf', 'poly'])})
# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# # PLOT HYPERPARAMETER SPACE
# my_pipe.optimizer.plot_evaluations()
# plt.show()
# my_pipe.optimizer.plot_objective()
# plt.show()

handler = ResultsTreeHandler(my_pipe.result_tree)

config_evaluations = handler.get_config_evaluations()
minimum_config_evaluations = handler.get_minimum_config_evaluations()

# handler.plot_optimizer_history('mean_squared_error', 'RGS 40 Eval (Scatter)', 'scatter',
#                                '/spm-data/Scratch/spielwiese_nils_winter/optimizer_history_random_grid_search_40_scatter.png')
handler.plot_optimizer_history('mean_squared_error', 'Scikit Optimize 40 Eval (Scatter)', 'scatter',
                               '/spm-data/Scratch/spielwiese_nils_winter/optimizer_history_scikit_optimize_40_scatter.png')
