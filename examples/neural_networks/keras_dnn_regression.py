from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical

# WE USE THE Boston_Housing SET FROM SKLEARN
X, y = load_boston(return_X_y=True)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_keras_regression_pipe',
                    optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=2),
                    inner_cv=KFold(n_splits=2),
                    verbosity=1,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

# attention: shape of hidden_layer_sizes == shape of activations. If you want to choose a function in every layer,
# grid_search eliminates combinations with len(hidden_layer_size) != len(activations).
# Check out: hidden_layer_sizes=[25, 10], activations=['tanh', 'relu']

# USE KERASDNNCLASSIFIER FOR CLASSIFICATION
my_pipe += PipelineElement('KerasDnnRegressor',
                           hyperparameters={'hidden_layer_sizes': Categorical([[18, 14], [30, 5]]),
                                            'dropout_rate': Categorical([0.01, 0.2])},
                           activations='relu',
                           epochs=50,
                           nn_batch_size=64,
                           verbosity=0)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
