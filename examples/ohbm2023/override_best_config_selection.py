from sklearn.datasets import load_diabetes
from photonai import RegressionPipe, PipelineElement
import numpy as np


def best_config_selector(list_of_non_failed_configs, metric, fold_operation, maximize_metric):
    # list_of_config_vals = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]
    #
    # if maximize_metric:
    #     # max metric
    #     best_config_metric_nr = np.argmax(list_of_config_vals)
    # else:
    #     # min metric
    #     best_config_metric_nr = np.argmin(list_of_config_vals)
    #
    # best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

    return list_of_non_failed_configs[0]


my_pipe = RegressionPipe('diabetes',
                         add_estimator=False,
                         select_best_config_delegate=best_config_selector)

my_pipe += PipelineElement('SVC')
# load data and train
X, y = load_diabetes(return_X_y=True)
my_pipe.fit(X, y)
