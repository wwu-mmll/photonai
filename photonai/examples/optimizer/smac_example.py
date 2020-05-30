from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch
from photonai.optimization import FloatRange, Categorical, IntegerRange

# WE USE THE BOSTON HOUSING DATA FROM SKLEARN
X, y = load_boston(return_X_y=True)

scenario_dict = {"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": 30,  # maximum function evaluations
                             "deterministic": "true",
                             "wallclock_limit": 60*2
                             }

# DESIGN YOUR PIPELINE
settings = OutputSettings(project_folder='./tmp/')
my_pipe = Hyperpipe('smac_example',
                    optimizer='smac',  # which optimizer PHOTON shall use, in this case smac
                    optimizer_params={'scenario_dict': scenario_dict},
                    metrics=['mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=ShuffleSplit(n_splits=1, test_size=0.2),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    output_settings=settings)


# ADD ELEMENTS TO YOUR PIPELINE
# first normalize all features
my_pipe.add(PipelineElement('StandardScaler'))
# then do feature selection using a PCA, specify which values to try in the hyperparameter search
my_pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 10)}, test_disabled=True)

switch = Switch("Test_Switch")
# engage and optimize SVR
# linspace and logspace is converted to uniform and log-uniform priors in skopt
switch += PipelineElement('SVR', hyperparameters={'C': FloatRange(0, 10, range_type='linspace'),
                                                   'epsilon': FloatRange(0, 0.0001, range_type='linspace'),
                                                   'tol': FloatRange(1e-4, 1e-2, range_type='linspace'),
                                                   'kernel': Categorical(['linear', 'rbf', 'poly'])})

switch += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': Categorical([10, 20])})

my_pipe += switch

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
