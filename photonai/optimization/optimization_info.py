import numpy as np
import json
from photonai.optimization import GridSearchOptimizer, TimeBoxedRandomGridSearchOptimizer, RandomGridSearchOptimizer, \
    SkOptOptimizer, RandomSearchOptimizer, SMACOptimizer
from photonai.optimization.switch_optimizer.meta_optimizer import MetaHPOptimizer
from photonai.processing.metrics import Scorer
from photonai.photonlogger.logger import logger
from photonai.processing.results_structure import MDBHelper, FoldOperations


class Optimization:

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                            'random_grid_search': RandomGridSearchOptimizer,
                            'timeboxed_random_grid_search': TimeBoxedRandomGridSearchOptimizer,
                            'sk_opt': SkOptOptimizer,
                            'smac': SMACOptimizer,
                            'random_search': RandomSearchOptimizer,
                            'switch': MetaHPOptimizer}

    def __init__(self, optimizer_input, optimizer_params,
                 metrics, best_config_metric, performance_constraints):

        self._optimizer_input = ''
        self.optimizer_input_str = optimizer_input
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self._best_config_metric = ''
        self.maximize_metric = True
        self.best_config_metric = best_config_metric
        self.performance_constraints = performance_constraints



    @property
    def best_config_metric(self):
        return self._best_config_metric

    @best_config_metric.setter
    def best_config_metric(self, value):
        self._best_config_metric = value
        if isinstance(self.best_config_metric, str):
            self.maximize_metric = Scorer.greater_is_better_distinction(self.best_config_metric)

    @property
    def optimizer_input_str(self):
        return self._optimizer_input

    @optimizer_input_str.setter
    def optimizer_input_str(self, value):
        if isinstance(value, str):
            if value not in self.OPTIMIZER_DICTIONARY:
                raise ValueError("Optimizer " + value + " not supported right now.")
        self._optimizer_input = value

    def sanity_check_metrics(self):

        if self.best_config_metric is not None:
            if isinstance(self.best_config_metric, list):
                warning_text = "Best Config Metric must be a single metric given as string, no list. " \
                               "PHOTON chose the first one from the list of metrics to calculate."

                self.best_config_metric = self.best_config_metric[0]
                logger.warning(warning_text)
                raise Warning(warning_text)
            elif not isinstance(self.best_config_metric, str):
                self.best_config_metric = Scorer.register_custom_metric(self.best_config_metric)

            if self.metrics is None:
                # if only best_config_metric is given, copy if to list of metrics
                self.metrics = [self.best_config_metric]
            else:
                # if best_config_metric is not given in metrics list, copy it to list
                if self.best_config_metric not in self.metrics:
                    self.metrics.append(self.best_config_metric)

        if self.metrics is not None and len(self.metrics) > 0:
            for i in range(len(self.metrics)):
                if not isinstance(self.metrics[i], str):
                    self.metrics[i] = Scorer.register_custom_metric(self.metrics[i])
            self.metrics = list(filter(None, self.metrics))
        else:
            error_msg = "No metrics were chosen. Please choose metrics to quantify your performance and set " \
                        "the best_config_metric so that PHOTON which optimizes for"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.best_config_metric is None and self.metrics is not None and len(self.metrics) > 0:
            self.best_config_metric = self.metrics[0]
            warning_text = "No best config metric was given, so PHOTON chose the first in the list of metrics as " \
                           "criteria for choosing the best configuration."
            logger.warning(warning_text)
            raise Warning(warning_text)

    def get_optimizer(self):
        if isinstance(self.optimizer_input_str, str):
            # instantiate optimizer from string
            optimizer_class = self.OPTIMIZER_DICTIONARY[self.optimizer_input_str]
            optimizer_instance = optimizer_class(**self.optimizer_params)
            return optimizer_instance
        else:
            # Todo: check if object has the right interface
            return self.optimizer_input_str

    def get_optimum_config(self, tested_configs, fold_operation=FoldOperations.MEAN):
        """
        Looks for the best configuration according to the metric with which the configurations are compared -> best config metric
        :param tested_configs: the list of tested configurations and their performances
        :return: MDBConfiguration that has performed best
        """

        list_of_config_vals = []
        list_of_non_failed_configs = [conf for conf in tested_configs if not conf.config_failed]

        if len(list_of_non_failed_configs) == 0:
            raise Warning("No Configs found which did not fail.")
        try:

            if len(list_of_non_failed_configs) == 1:
                best_config_outer_fold = list_of_non_failed_configs[0]
            else:
                for config in list_of_non_failed_configs:
                    list_of_config_vals.append(
                        MDBHelper.get_metric(config, fold_operation, self.best_config_metric, train=False))

                if self.maximize_metric:
                    # max metric
                    best_config_metric_nr = np.argmax(list_of_config_vals)
                else:
                    # min metric
                    best_config_metric_nr = np.argmin(list_of_config_vals)

                best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

            # inform user
            logger.debug('Optimizer metric: ' + self.best_config_metric + '\n' +
                         '   --> Maximize metric: ' + str(self.maximize_metric))

            logger.info('Number of tested configurations: ' + str(len(tested_configs)))
            logger.photon_system_log(
                '---------------------------------------------------------------------------------------------------------------')
            logger.photon_system_log('BEST_CONFIG ')
            logger.photon_system_log(
                '---------------------------------------------------------------------------------------------------------------')
            logger.photon_system_log(json.dumps(best_config_outer_fold.human_readable_config, indent=4,
                                                sort_keys=True))

            return best_config_outer_fold
        except BaseException as e:
            logger.error(str(e))

    def get_optimum_config_outer_folds(self, outer_folds):
        list_of_scores = list()
        for outer_fold in outer_folds:
            metrics = outer_fold.best_config.best_config_score.validation.metrics
            list_of_scores.append(metrics[self.best_config_metric])

        if self.maximize_metric:
            # max metric
            best_config_metric_nr = np.argmax(list_of_scores)
        else:
            # min metric
            best_config_metric_nr = np.argmin(list_of_scores)

        best_config = outer_folds[best_config_metric_nr].best_config
        return best_config