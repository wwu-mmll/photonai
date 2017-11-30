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
                    inner_cv=KFold(n_splits=5, shuffle=True), best_config_metric='accuracy',
                    metrics=['accuracy', 'precision', 'recall', "f1_score"], logging=True, eval_final_performance=True, verbose=2)

manager.add(PipelineElement.create('standard_scaler', test_disabled=True))

# nn = PipelineElement.create('kdnn', hyperparameters={'hidden_layer_sizes': [[5, 3]]})
svm = PipelineElement.create('svc', hyperparameters={'C': [0.5, 1]}, kernel='rbf')
# manager.add(PipelineSwitch('final_estimator', [nn, svm]))

manager.add(svm)
# manager.add(nn)
manager.fit(X, y)

result_tree = manager.result_tree

# get metrics for no outer cv, for inner fold 1 and for default config:
metrics = result_tree.get_metrics_for_inner_cv(outer_cv_fold=0, inner_cv_fold=0, config_nr=0)

all_metrics = result_tree.get_all_metrics()

# get best config of outer cv fold 1:
best_config = result_tree.get_best_config_for(outer_cv_fold=0)

# performance of best config of outer cv fold 1 for TEST DATA:
# -> INCLUDING: metrics, y_true and y_predicted
# -> on this object you can also call helper functions such as roc_curve (which is not tested yet)
best_config_performance_test = result_tree.get_best_config_performance_for(outer_cv_fold=0)

# performance of best config of outer cv fold 1 for TRAIN DATA:
best_config_performance_train = result_tree.get_best_config_performance_for(outer_cv_fold=0, train_data=True)

# iterate all tested configuration for outer fold 1:
tested_configs = result_tree.get_tested_configurations_for(outer_cv_fold=0)


# THE END
debugging = True

from Framework import LogExtractor
log_ex = LogExtractor.LogExtractor(result_tree)
log_ex.extract_csv("test.csv")
