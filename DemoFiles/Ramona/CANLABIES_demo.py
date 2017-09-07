import numpy as np
from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch, PipelineStacking
# from Helpers.DataIntuition import show_pca
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_breast_cancer

# LOAD DATA
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
# show_pca(X, y)
print(np.sum(y)/len(y))

# BUILD PIPELINE
manager = Hyperpipe('test_manager',
                    optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1},
                    outer_cv=ShuffleSplit(test_size=0.2, n_splits=3),
                    inner_cv=KFold(n_splits=3, shuffle=True),
                    metrics=['accuracy', 'precision', 'recall'], logging=True, eval_final_performance=True)

manager.add(PipelineElement.create('standard_scaler', test_disabled=True))

# nn = PipelineElement.create('kdnn', hyperparameters={'hidden_layer_sizes': [[5, 3]]})
svm = PipelineElement.create('svc', hyperparameters={'C': [0.5, 1]}, kernel='rbf')
# manager.add(PipelineSwitch('final_estimator', [nn, svm]))

manager.add(svm)
# manager.add(nn)
manager.fit(X, y)


# THE END
debugging = True
