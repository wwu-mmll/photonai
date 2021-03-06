from enum import Enum
import numpy as np
import inspect
from photonai.processing.metrics import Scorer
from photonai.photonlogger.logger import logger


class PhotonBaseConstraint:
    """
    The PHOTON base interface for any performance constraints that could speed up hyperparameter search.
    After a particular configuration is tested in one fold, the performance constraint objects are called to
    evaluate if the configuration is promising. If not, further testing in other folds is skipped to increase speed.
    """

    ENUM_STRATEGY = Enum("strategy", "first all mean")

    def __init__(self, strategy='first', metric='', threshold: float = None, margin: float = 0, **kwargs):
        self._metric = None
        self._greater_is_better = None
        self._strategy = None

        # with setting property we automatically find greater_is_better
        self.metric = metric
        self.threshold = threshold
        self.margin = margin
        self.strategy = strategy

    @property
    def strategy(self):
        """
        Getter for attribute strategy.
        :return:
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        """
        Setter for strategy. Checks if strategy is supported.
        :param value: String
        :return:
        """
        try:
            self._strategy = PhotonBaseConstraint.ENUM_STRATEGY[value]
        except KeyError:
            raise KeyError("Your strategy: " + str(value) + " is not supported yet. Please use one of " +
                           str([x.name for x in PhotonBaseConstraint.ENUM_STRATEGY]))

    @property
    def metric(self):
        """
        Getter for attribute metric.
        :return:
        """
        return self._metric

    @metric.setter
    def metric(self, value):
        """
        Setter for attribute metric.
        :param value: metric value
        :return:
        """
        try:
            self._metric = value
            self._greater_is_better = Scorer.greater_is_better_distinction(self._metric)
        except NameError:
            self._metric = "unknown"
            logger.warning("Your metric is not supported. Performance constraints are constantly False.")

    def shall_continue(self, config_item):
        """
        Function to evaluate if the constraint is reached.
        If it returns True, the testing of the configuration is continued.
        If it returns False, further testing of the configuration is skipped
        to increase speed of the hyperparameter search.

        Parameters
        ----------
        * 'config_item' [MDBConfig]:
            All performance metrics and other scoring information for all configuration's performance.
            Can be used to evaluate if the configuration has any potential to serve the model's learning task.
        """
        if self.metric == "unknown":
            logger.warning("The metric is not known. Please check the metric: " + self.metric + ". " +
                        "Performance constraints are constantly True.")
            return True
        if self.metric not in config_item.inner_folds[0].validation.metrics:
            logger.warning("The metric is not calculated. Please insert " + self.metric + " to Hyperpipe.metrics. " +
                        "Performance constraints are constantly False.")
            return False
        if self._greater_is_better:
            if self.strategy.name == 'first':
                if config_item.inner_folds[0].validation.metrics[self.metric] < self.threshold:
                    return False
            elif self.strategy.name == 'all':
                if any(item < self.threshold for item in [x.validation.metrics[self.metric]
                                                          for x in config_item.inner_folds]):
                    return False
            elif self.strategy.name == 'mean':
                if np.mean([x.validation.metrics[self.metric] for x in config_item.inner_folds]) < self.threshold:
                    return False
            return True
        else:
            if self.strategy.name == 'first':
                if config_item.inner_folds[0].validation.metrics[self.metric] > self.threshold:
                    return False
            elif self.strategy.name == 'all':
                if any(item > self.threshold for item in [x.validation.metrics[self.metric]
                                                          for x in config_item.inner_folds]):
                    return False
            elif self.strategy.name == 'mean':
                if np.mean([x.validation.metrics[self.metric] for x in config_item.inner_folds]) > self.threshold:
                    return False
            return True

    def copy_me(self):
        """
        Copy self object.
        :return:
        """
        new_me = type(self)(metric=self.metric)
        signature = inspect.getfullargspec(self.__init__)[0]
        for attr in signature:
            if not attr == 'self' and hasattr(self, attr) and attr != 'strategy':
                setattr(new_me, attr, getattr(self, attr))
            elif attr == 'strategy':
                setattr(new_me, attr, getattr(self, attr).name)
        return new_me


class MinimumPerformance(PhotonBaseConstraint):
    """
    Tests if a configuration performs better than a given limit for a particular metric.

    Example
    -------
    MinimumPerformance('accuracy', 0.96) tests if the configuration has at least a performance of 0.96 in
    (the) [first, all, mean] fold(s).
    If not further testing of the configuration is skipped, as it is regarded as not promising enough.
    """

    def __init__(self, metric: str = '', threshold: float = 1., strategy='first'):
        super(MinimumPerformance, self).__init__(strategy=strategy, metric=metric, threshold=threshold)


class DummyPerformance(PhotonBaseConstraint):
    """
    Tests if a configuration performs better than a given limit for a particular metric.

    Example
    -------
    DummyPerformance('accuracy', 0.1) tests if the configuration has at least a 10% better performance than the dummy
    estimator. Distinguish between [first, all, mean] fold(s).
    If not further testing of the configuration is skipped, as it is regarded as not promising enough.
    """

    def __init__(self, metric: str = '', margin: float = 0, strategy='first'):
        super(DummyPerformance, self).__init__(strategy=strategy, metric=metric, margin=margin)

    def set_dummy_performance(self, dummy_result):
        """
        Set threshold with margin and dummy_performance value
        :param dummy_result: MDBScoreInformation type
        :return:
        """
        self.threshold = dummy_result.validation.metrics[self.metric] + self.margin

    def copy_me(self):
        """
        Copy self object. Appending threshold to super.copy_me().
        :return: DummyPerformance
        """
        new_me = super(DummyPerformance, self).copy_me()
        if "threshold" in self.__dict__.keys():
            new_me.threshold = self.threshold
        return new_me


class BestPerformance(PhotonBaseConstraint):
    """
    BestPerformance decides in every fold: challenger works better than incumbent
    true: eval next fold, false: eval next config
    better in all inner_folds: incumbent = challenger.
    """

    def __init__(self, metric: str = '', strategy: str = 'mean'):
        """
        :param metric:
        :param strategy:
        :param fold_start:
        """
        super(BestPerformance, self).__init__(strategy=strategy, metric=metric)
        self.threshold = None
        self.config_items = {}
        self.required_folds = 0
        self.run = 0

    def shall_continue(self, config_item):
        """
        Function to evaluate if the constraint is reached.
        If it returns True, the testing of the configuration is continued.
        If it returns False, further testing of the configuration is skipped
        to increase speed of the hyperparameter search.

        Parameters
        ----------
        * 'config_item' [MDBConfig]:
            All performance metrics and other scoring information for all configuration's performance.
            Can be used to evaluate if the configuration has any potential to serve the model's learning task.
        """
        self.required_folds = max(self.required_folds, len(self.eval_config_entries(config_item)))
        self.config_items[str(config_item.config_dict)] = self.eval_config_entries(config_item)

        # first run default 10 configs
        if self.run < 10 * self.required_folds -1:
            self.run += 1
            return True

        # calc threshold for the first time over all available configs.
        if self.threshold is None:
            if self._greater_is_better:
                self.threshold = max([np.mean(x) for x in self.config_items.values()])
            else:
                self.threshold = min([np.mean(x) for x in self.config_items.values()])

        # calc std between inner_folds and average it over all configs [std only for multiple-run-configs]
        std = np.mean([np.std(x) for x in self.config_items.values() if len(self.config_items.values())>1])
        challenger = self.eval_config_entries(config_item)
        if self._greater_is_better:
            if np.mean(challenger) > self.threshold - std:
                if np.mean(challenger) > self.threshold and len(challenger) == self.required_folds:
                    self.threshold = np.mean(challenger)
                return True
        else:
            if np.mean(challenger) < self.threshold + std:
                if np.mean(challenger) < self.threshold and len(challenger) == self.required_folds:
                    self.threshold = np.mean(challenger)
                return True
        return False

    def copy_me(self):
        """
        Copy self object. Appending threshold to super.copy_me().
        :return: BestPerformance
        """
        new_me = super(BestPerformance, self).copy_me()
        if "threshold" in self.__dict__.keys():
            new_me.threshold = self.threshold
        return new_me

    def eval_config_entries(self, config_item):
        return [x.validation.metrics[self.metric] for x in config_item.inner_folds]
