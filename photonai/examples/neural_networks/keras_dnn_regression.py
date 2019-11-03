from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from photonai.investigator import Investigator
from photonai.base import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization import Categorical

# WE USE THE Boston_Housing SET FROM SKLEARN
X, y = load_boston(return_X_y=True)


settings =

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_keras_regression_pipe',
                    optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error'],
                    best_config_metric='mean_squared_error',
                    outer_cv=KFold(n_splits=2),
                    inner_cv=KFold(n_splits=2),
                    verbosity=1,
                    output_settings=OutputSettings(project_folder='./tmp/'))


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

# attention: hidden_layer count == activation size. So if you want to choose a function in every layer,
# grid_search does not forbid combinations with len(hidden_layer_size) != len(activations)

# USE KERASDNNCLASSIFIER FOR CLASSIFICATION
my_pipe += PipelineElement('KerasDnnRegressor',
                           hyperparameters={'hidden_layer_sizes': Categorical([[10, 8, 4], [20, 5]]),
                                            'dropout_rate': Categorical([0.5, [0.5, 0.2]])
                                            },
                           activations='relu',
                           epochs=5,
                           batch_size=32,
                           verbosity=1)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)

debug = True
Investigator.show(my_pipe)
