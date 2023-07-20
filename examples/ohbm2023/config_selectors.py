import numpy as np


class DefaultConfigSelector:

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):

        list_of_config_vals = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]

        if maximize_metric:
            # max metric
            best_config_metric_nr = np.argmax(list_of_config_vals)
        else:
            # min metric
            best_config_metric_nr = np.argmin(list_of_config_vals)

        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold

