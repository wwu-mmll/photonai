from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import FloatRange, Categorical
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_boston
import matplotlib.pylab as plt

# WE USE THE BOSTON HOUSING DATA FROM SKLEARN
X, y = load_boston(True)



# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('skopt_example',
                    optimizer='sk_opt',  # which optimizer PHOTON shall use, in this case sk_opt
                    optimizer_params={'num_iterations': 50, 'acq_func': 'LCB', 'acq_func_kwargs': {'kappa': 1.96}},
                    metrics=['mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                    inner_cv=KFold(n_splits=3),
                    verbosity=0)



# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

# engage and optimize SVR
# linspace and logspace is converted to uniform and log-uniform priors in skopt
my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(1e-3, 100, range_type='logspace'),
                                                   'epsilon': FloatRange(1e-3, 10, range_type='logspace'),
                                                   'tol': FloatRange(1e-4, 1e-2, range_type='linspace'),
                                                   'kernel': Categorical(['linear', 'rbf', 'poly'])})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

# PLOT HYPERPARAMETER SPACE
my_pipe.optimizer.plot_evaluations()
plt.show()
my_pipe.optimizer.plot_objective()
plt.show()
