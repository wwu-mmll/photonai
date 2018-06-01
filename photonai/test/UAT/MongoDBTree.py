import sys
sys.path.append("..")
import numpy as np
from Framework.PhotonBase import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import Lasso
#  -----------> calculate something ------------------- #

# LOAD DATA
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
print(np.sum(y)/len(y))

from pymodm import connect
connect("mongodb://localhost:27017/photon_db")

# BUILD PIPELINE
manager = Hyperpipe('test_manager',
                    optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1},
                    outer_cv=ShuffleSplit(test_size=0.2, n_splits=3),
                    inner_cv=KFold(n_splits=10, shuffle=True), best_config_metric='accuracy',
                    metrics=['accuracy', 'precision', 'recall', "f1_score"],
                    logging=False, eval_final_performance=True,
                    calculate_metrics_across_folds=True,
                    verbose=2)

manager.add(PipelineElement.create('standard_scaler', test_disabled=True))
manager += PipelineElement.create('pca', hyperparameters={'n_components': [None, 1, 10000]})
# tmp_lasso = Lasso()
# manager.add(PipelineElement.create('SelectModelWrapper', estimator_obj=tmp_lasso))

svm = PipelineElement.create('svc', hyperparameters={'C': [0.5, 1], 'kernel': ['linear']})
manager.add(svm)
manager.fit(X, y)

#  -----------> Result Tree generated ------------------- #
result_tree = manager.result_tree
# result_tree.write_to_db()

# THE END
debugging = True
