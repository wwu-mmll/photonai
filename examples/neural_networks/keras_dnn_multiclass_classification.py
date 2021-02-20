from sklearn.datasets import load_digits
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import Categorical

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = load_digits(n_class=5, return_X_y=True)

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_keras_multiclass_pipe',
                    optimizer='grid_search',
                    optimizer_params={},
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    outer_cv=KFold(n_splits=2),
                    inner_cv=KFold(n_splits=2),
                    verbosity=1,
                    project_folder='./tmp/')


# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

# attention: shape of hidden_layer_sizes == shape of activations. If you want to choose a function in every layer,
# grid_search eliminates combinations with len(hidden_layer_size) != len(activations).
# Check out: hidden_layer_sizes=[25, 10], activations=['tanh', 'relu']
my_pipe += PipelineElement('KerasDnnClassifier',
                           hyperparameters={'hidden_layer_sizes': Categorical([[20, 10, 5], [10, 8, 4]]),
                                            'dropout_rate': Categorical([0.5, [0.5, 0.5, 0.5]])},
                           activations='relu',
                           nn_batch_size=32,
                           epochs=50,
                           multi_class=True,
                           verbosity=0)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
