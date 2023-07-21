import numpy as np


class BaseConfigSelector:

    def prepare_metrics(self, list_of_non_failed_configs, metric):
        classification_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1_score', 'matthews_corrcoef']
        regression_metrics = ['mean_absolute_error', 'mean_squared_error', 'explained_variance']

        # right now we can only do this ugly hack, sorry!
        is_it_classification = metric in classification_metrics
        all_metric_mean = {}
        all_metric_std = {}
        metric_list = classification_metrics if is_it_classification is True else regression_metrics
        for m in metric_list:
            all_metric_mean[m] = [c.get_test_metric(m, "mean") for c in list_of_non_failed_configs]
            all_metric_std[m] = [c.get_test_metric(m, "std") for c in list_of_non_failed_configs]
        # -----------------
        return all_metric_mean, all_metric_std, metric_list, is_it_classification


class DefaultConfigSelector(BaseConfigSelector):

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):

        all_metrics_mean, all_metrics_std = self.prepare_metrics(list_of_non_failed_configs, metric)
        best_config_metric_values = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]

        if maximize_metric:
            # max metric
            best_config_metric_nr = np.argmax(best_config_metric_values)
        else:
            # min metric
            best_config_metric_nr = np.argmin(best_config_metric_values)

        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold


class RandomConfigSelector(BaseConfigSelector):

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):
        
        best_config_metric_nr = np.random.randint(0, len(list_of_non_failed_configs))
        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold
    
class RankingAgreementConfigSelector(BaseConfigSelector):
     
    def rank_metrics(self, metric_values, maximize_metric):

        # Sort the metric values in descending (maximization) or ascending (minimization) order.
        if maximize_metric:
            sorted_indices = np.argsort(metric_values)[::-1]
        else:
            sorted_indices = np.argsort(metric_values)

        ranks = np.empty(len(metric_values), dtype=int)
        ranks[sorted_indices] = np.arange(1, len(metric_values) + 1)

        return ranks
     
    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):
        
        all_metrics_mean, all_metrics_std, metric_list, is_it_classification = self.prepare_metrics(list_of_non_failed_configs, metric)
        
        metric_ranks = np.array([]) 

        if is_it_classification: 
            current_maximize_metrics = [True, True, True, True, True]
        else:
            current_maximize_metrics = [False, False, True]

        for i, current_metric in enumerate(metric_list) :
        
            best_config_metric_values = [c.get_test_metric(current_metric, fold_operation) for c in list_of_non_failed_configs]
        
            # Rank the metric values
            metric_ranks = np.append(metric_ranks, self.rank_metrics(best_config_metric_values, current_maximize_metrics[i]))

        total_ranks = np.sum(metric_ranks)
        
        # Select the configuration with the best rank sum
        best_config_metric_nr = np.argmin(total_ranks)
        
        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold

