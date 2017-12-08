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
                    outer_cv=ShuffleSplit(test_size=0.2, n_splits=1),
                    inner_cv=KFold(n_splits=10, shuffle=True), best_config_metric='accuracy',
                    metrics=['accuracy', 'precision', 'recall', "f1_score"], logging=True, eval_final_performance=True, verbose=2)

manager.add(PipelineElement.create('standard_scaler', test_disabled=True))

# nn = PipelineElement.create('kdnn', hyperparameters={'hidden_layer_sizes': [[5, 3]]})
svm = PipelineElement.create('svc', hyperparameters={'C': [0.5, 1], 'kernel': ['linear', 'rbf']})
# manager.add(PipelineSwitch('final_estimator', [nn, svm]))

manager.add(svm)
# manager.add(nn)
manager.fit(X, y)

result_tree = manager.result_tree

best_config_outer_cv_0 = result_tree.get_best_config_for(outer_cv_fold=0)
feature_weights_best_config_outer_cv_0 = best_config_outer_cv_0.fold_list[0].train.feature_importances_
feature_weights_any_config = result_tree.get_feature_importances_for_inner_cv(outer_cv_fold=0, inner_cv_fold=1, config_nr=0)

# inverse_transformed_feature_importances = manager.inverse_transform_pipeline(best_config_outer_cv_0.config_dict, X, y,
#                                                               feature_weights_best_config_outer_cv_0)

# get metrics for no outer cv, for inner fold 1 and for default config:
metrics = result_tree.get_metrics_for_inner_cv(outer_cv_fold=0, inner_cv_fold=0, config_nr=0)

all_metrics = result_tree.get_all_metrics()

# get best config of outer cv fold 1:
best_config = result_tree.get_best_config_for(outer_cv_fold=0)


predictions_of_inner_fold = result_tree.get_predictions_for_inner_cv()

plot_config_performances = result_tree.plot_config_performances_for_outer_fold()

# THE END
debugging = True

# from Framework import LogExtractor
# log_ex = LogExtractor.LogExtractor(result_tree)
# log_ex.extract_csv("test.csv")
