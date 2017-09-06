import numpy as np
from sklearn.model_selection import ShuffleSplit

import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe

#X = np.reshape(X, (28,28,70000))

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# y_train = tfu.oneHot(y_train)
# y_test = tfu.oneHot(y_test)
# X_train = np.reshape(X_train, (60000,28,28,1))
# X_test = np.reshape(X_test, (10000,28,28,1))

X_train = np.random.rand(100,224,224,3)
y_train = np.random.randint(0,10,100)
y_train = tfu.binary_to_one_hot(y_train)
cv = ShuffleSplit(n_splits=1,test_size=0.2, random_state=23)


#cv = KFold(n_splits=5, random_state=23)
my_pipe = Hyperpipe('mnist_siamese_net', optimizer='grid_search',
                    metrics=['categorical_accuracy'], best_config_metric='categorical_accuracy',
                    inner_cv=cv,
                    outer_cv=cv,
                    eval_final_performance=True, verbose=2)
#my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('PretrainedCNNClassifier', {'input_shape': [(224,224,3)],'target_dimension': [10],  'nb_epoch':[100], 'size_additional_layer':[100], 'freezing_point':[0]})
my_pipe.fit(X_train,y_train)
