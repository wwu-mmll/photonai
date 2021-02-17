from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

from photonai.base import Hyperpipe, PipelineElement

# WE USE THE BREAST CANCER SET FROM SKLEARN
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

callbacks = [es]

# ADD ELEMENTS TO YOUR PIPELINE
my_pipe.add(PipelineElement('StandardScaler'))

# attention: shape of hidden_layer_sizes == shape of activations. If you want to choose a function in every layer,
# grid_search permits combinations with len(hidden_layer_size) != len(activations).
# Check out: hidden_layer_sizes=[25, 10], activations=['tanh', 'relu']
my_pipe += PipelineElement('KerasDnnClassifier',
                           hidden_layer_sizes=[10],
                           activations='relu',
                           nn_batch_size=128,
                           callbacks=callbacks,
                           epochs=50,
                           multi_class=True,
                           verbosity=1)

# NOW TRAIN YOUR PIPELINE
my_pipe.fit(X, y)
