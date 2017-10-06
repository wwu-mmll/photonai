from sklearn.datasets import fetch_mldata
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe
from Logging.Logger import Logger

mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
y_one_hot = tfu.binary_to_one_hot(y)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.01)
for _, test_index in sss.split(X, y):
    X_small = X[test_index]
    y_small = y[test_index]
y_small = tfu.binary_to_one_hot(y_small)


cv = ShuffleSplit(n_splits=1,test_size=0.2, random_state=23)
#cv = KFold(n_splits=5, random_state=23)
my_pipe = Hyperpipe('mnist_siamese_net', optimizer='grid_search',
                    metrics=['categorical_accuracy'], best_config_metric='categorical_accuracy',
                    inner_cv=cv,
                    outer_cv=cv,
                    eval_final_performance=True, verbose=2)
my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('KerasDNNClassifier', {'hidden_layer_sizes': [[64,64]],'target_dimension': [10], 'dropout_rate': [0.5], 'nb_epoch':[100]})
my_pipe.fit(X_small,y_small)

# try predict_proba functionality
Logger().info('Predictions (Probs): {0}'.format(my_pipe.predict_proba(X_small)))