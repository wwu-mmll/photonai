from enum import Enum
import numpy as np
import inspect
from photonai.processing.metrics import Scorer


class PhotonBaseConstraint:
    """
    The PHOTON base interface for any performance constraints that could speed up hyperparameter search.
    After a particular configuration is tested in one fold, the performance constraint objects are called to
    evaluate if the configuration is promising. If not, further testing in other folds is skipped to increase speed.
    """

    ENUM_STRATEGY = Enum("strategy", "first all mean")

    def __init__(self, strategy='first', metric='', threshold: float=None, margin: float=0, **kwargs):
        self._metric = None
        self._greater_is_better = None
        self._strategy = ''

        # with setting property we automatically find greater_is_better
        self.strategy = strategy
        self.metric = metric
        self.threshold = threshold
        self.margin = margin

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        try:
            self._strategy = PhotonBaseConstraint.ENUM_STRATEGY[value]
        except KeyError:
            raise ValueError("Your strategy: " + str(value) + " is not supported yet. Please use one of " +
                             str([x.name for x in PhotonBaseConstraint.ENUM_STRATEGY]))

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value
        self._greater_is_better = Scorer.greater_is_better_distinction(self._metric)

    def shall_continue(self, config_item):
        """
        Function to evaluate if the constraint is reached.
        If it returns True, the testing of the configuration is continued.
        If it returns False, further testing of the configuration is skipped to increase speed of the hyperparameter search.

        Parameters
        ----------
        * 'config_item' [MDBConfig]:
            All performance metrics and other scoring information for all configuration's performance.
            Can be used to evaluate if the configuration has any potential to serve the model's learning task.
        """
        if self._greater_is_better:
            if self.strategy.name == 'first':
                if config_item.inner_folds[0].validation.metrics[self.metric] < self.threshold:
                    return False
            elif self.strategy.name == 'all':
                if all(item > self.threshold for item in [x.validation.metrics[self.metric] for x in config_item.inner_folds]):
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
                if all(item > self.threshold for item in [x.validation.metrics[self.metric] for x in config_item.inner_folds]):
                    return False
            elif self.strategy.name == 'mean':
                if np.mean([x.validation.metrics[self.metric] for x in config_item.inner_folds]) > self.threshold:
                    return False
            return True

    def copy_me(self):
        new_me = type(self)()
        signature = inspect.getfullargspec(self.__init__)[0]
        for attr in signature:
            if not attr == 'self' and hasattr(self, attr):
                setattr(new_me, attr, getattr(self, attr))
        return new_me


class MinimumPerformance(PhotonBaseConstraint):
    """
    Tests if a configuration performs better than a given limit for a particular metric.

    Example
    -------
    MinimumPerformance('accuracy', 0.96) tests if the configuration has at least a performance of 0.96 in (the) [first, all, mean] fold(s).
    If not further testing of the configuration is skipped, as it is regarded as not promising enough.
    """

    def __init__(self, metric: str='', threshold: float=1, strategy='first'):
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

    def __init__(self, metric: str='', margin: float =1., strategy='first'):
        super(DummyPerformance, self).__init__(strategy=strategy, metric=metric, margin=margin)

    def set_dummy_performance(self, dummy_result):
        self.threshold = dummy_result.validation.metrics[self.metric]+self.margin

