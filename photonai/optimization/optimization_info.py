import numpy as np
from photonai.optimization import GridSearchOptimizer, RandomGridSearchOptimizer, \
    SkOptOptimizer, RandomSearchOptimizer, SMACOptimizer, NevergradOptimizer
from photonai.optimization.switch_optimizer.meta_optimizer import MetaHPOptimizer
from photonai.processing.metrics import Scorer
from photonai.photonlogger.logger import logger


class Optimization:

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                            'random_grid_search': RandomGridSearchOptimizer,
                            'sk_opt': SkOptOptimizer,
                            'smac': SMACOptimizer,
                            'random_search': RandomSearchOptimizer,
                            'nevergrad': NevergradOptimizer,
                            'switch': MetaHPOptimizer}

    def __init__(self, optimizer_input, optimizer_params,
                 metrics, best_config_metric, performance_constraints):

        self._optimizer_input = ''
        self.optimizer_input_str = optimizer_input
        self.optimizer_params = optimizer_params
        self._best_config_metric = ''
        self.performance_constraints = performance_constraints
        self.metrics = None
        self.maximize_metric = None
        self.best_config_metric = None
        self.sanity_check_metrics(metrics, best_config_metric)

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

    def sanity_check_metrics(self, metrics, best_config_metric):

        # first of all register all custom elements, if any
        if best_config_metric is not None and not isinstance(best_config_metric, str) \
                and not isinstance(best_config_metric, list):
            best_config_metric = Scorer.register_custom_metric(best_config_metric)

        if metrics is not None and len(metrics) > 0:
            for i in range(len(metrics)):
                if not isinstance(metrics[i], str):
                    metrics[i] = Scorer.register_custom_metric(metrics[i])

        if metrics is not None:
            metrics = list(filter(None, metrics))
        self.metrics = metrics
        self.best_config_metric = best_config_metric

        if self.metrics is None or len(self.metrics) == 0:
            if self.best_config_metric is None:
                error_msg = "No metrics were chosen. Please choose metrics to quantify your performance and set " \
                            "the best_config_metric so that PHOTON which optimizes for"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # if only best_config_metric is given, copy if to list of metrics
                self.metrics = [self.best_config_metric]

        if self.best_config_metric is not None:
            if isinstance(self.best_config_metric, list):
                warning_text = "Best Config Metric must be a single metric given as string, no list. " \
                               "PHOTON chose the first one from the list of metrics to calculate."

                self.best_config_metric = self.best_config_metric[0]
                logger.warning(warning_text)
                raise Warning(warning_text)

            # if best_config_metric is not given in metrics list, copy it to list
            if self.best_config_metric not in self.metrics:
                self.metrics.append(self.best_config_metric)

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