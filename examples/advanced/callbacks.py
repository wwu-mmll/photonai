from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from photonai.base import Hyperpipe, PipelineElement, CallbackElement


# DEFINE CALLBACK ELEMENT
def my_monitor(X, y=None, **kwargs):
    print(X.shape)

    # here is a useless statement where you can easily set a breakpoint
    # and do fancy developer stuff
    debug = True

my_pipe = Hyperpipe('monitoring_pipe',
                    optimizer='grid_search',
                    metrics=['mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=3),
                    inner_cv=KFold(n_splits=3),
                    verbosity=1,
                    project_folder='./tmp/')

my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SamplePairingClassification',
                           hyperparameters={'draw_limit': [500, 1000, 10000]},
                           distance_metric='euclidean',
                           generator='nearest_pair',
                           test_disabled=True)

# here we inspect the data after augmentation
my_pipe += CallbackElement("monitor", my_monitor)

my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': [10, 100]})

X, y = load_boston(return_X_y=True)
my_pipe.fit(X, y)
