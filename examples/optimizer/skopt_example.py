from sklearn.datasets import load_diabetes
from sklearn.model_selection import ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import FloatRange, Categorical

# WE USE THE DIABETES DATA FROM SKLEARN
X, y = load_diabetes(return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('skopt_example',
                    optimizer='sk_opt',  # which optimizer PHOTONAI shall use, in this case sk_opt
                    optimizer_params={'n_configurations': 25,
                                      'n_initial_points': 10,
                                      'base_estimator': 'GP',
                                      'initial_point_generator': 'grid',
                                      'acq_func': 'LCB',
                                      'acq_func_kwargs': {'kappa': 1.96}
                                      },
                    metrics=['mean_squared_error', 'mean_absolute_error'],
                    best_config_metric='mean_absolute_error',
                    outer_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                    inner_cv=ShuffleSplit(n_splits=3, test_size=0.3),
                    verbosity=0,
                    project_folder='./tmp/')

# ADD ELEMENTS TO YOUR PIPELINE
# first scale all features
my_pipe += PipelineElement('StandardScaler')

# engage and optimize SVR
# linspace and logspace are converted to uniform and log-uniform priors in skopt
my_pipe += PipelineElement('SVR', hyperparameters={'C': FloatRange(0.1, 100, range_type='linspace'),
                                                   'epsilon': FloatRange(1e-3, 10, range_type='logspace'),
                                                   'tol': FloatRange(1e-4, 1e-2, range_type='linspace'),
                                                   'kernel': Categorical(['linear', 'rbf', 'poly'])})

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
