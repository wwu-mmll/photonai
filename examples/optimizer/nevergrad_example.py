from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit
import nevergrad as ng

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import BooleanSwitch, FloatRange

X, y = load_boston(return_X_y=True)

# list of all available nevergrad optimizer
print(list(ng.optimizers.registry.values()))

my_pipe = Hyperpipe('nevergrad_optimization_pipe',
                    optimizer='nevergrad',
                    optimizer_params={'facade': 'NGO', 'n_configurations': 30},
                    metrics=['mean_squared_error', 'pearson_correlation', 'mean_absolute_error', 'explained_variance'],
                    best_config_metric='mean_squared_error',
                    outer_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                    inner_cv=ShuffleSplit(n_splits=3, test_size=0.2),
                    verbosity=0,
                    project_folder='./tmp/')

# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('PCA', n_components='mle')

my_pipe += PipelineElement('Ridge', hyperparameters={'alpha': FloatRange(0.1, 100),
                                                     'fit_intercept': BooleanSwitch(),
                                                     'normalize': BooleanSwitch()})

my_pipe.fit(X, y)
