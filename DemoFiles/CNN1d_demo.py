import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from Framework.PhotonBase import Hyperpipe, PipelineElement

dataset = load_breast_cancer()
X = dataset.data
# convnets need rgb color dimension
# just add additional dimension
X = np.reshape(X,(X.shape[0], X.shape[1], 1))
y = dataset.target


# test the complete hyperparameter search with KFold(n_splits=3)
manager = Hyperpipe('outer_man', KFold(n_splits=2), metrics=['accuracy'],
                    hyperparameter_search_cv_object=KFold(n_splits=3))


manager.add(PipelineElement.create('CNN1d', {'n_filters': [[16],[16,32]],
                                             'kernel_size': [3, 5],
                                             'dropout_rate': [0, 0.5]},
                                   target_dimension=2, stride=5,
                                   pooling_size=1, size_last_layer=15))
manager.fit(X, y)
