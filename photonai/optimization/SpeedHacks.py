from enum import Enum
import numpy as np
import inspect
enum_strategy = Enum("strategy", "first all mean")


class PhotonBaseConstraint:
    """
    The PHOTON base interface for any performance constraints that could speed up hyperparameter search.
    After a particular configuration is tested in one fold, the performance constraint objects are called to
    evaluate if the configuration is promising. If not, further testing in other folds is skipped to increase speed.
    """

    def __init__(self, *kwargs):
        pass

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
        pass

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

    def __init__(self,  metric: str='', smaller_than: float=1, strategy='first'):
        self.metric = metric
        self.smaller_than = smaller_than
        try:
            self.strategy = enum_strategy[strategy]
        except:
            raise AttributeError("Your strategy: "+str(strategy)+" is not supported yet. Please use one of "+str([x.name for x in enum_strategy]))

    def shall_continue(self, config_item):
        if self.strategy.name == 'first':
            if config_item.inner_folds[0].validation.metrics[self.metric] < self.smaller_than:
                return False
        elif self.strategy.name == 'all':
            if all(item > self.smaller_than for item in [x.validation.metrics[self.metric] for x in config_item.inner_folds]):
                return False
        elif self.strategy.name == 'mean':
            if np.mean([x.validation.metrics[self.metric] for x in config_item.inner_folds])  < self.smaller_than:
                return False
        return True


class DummyPerformance(PhotonBaseConstraint):
    """
    Tests if a configuration performs better than a given limit for a particular metric.

    Example
    -------
    DummyPerformance('accuracy', 0.1) tests if the configuration has at least a 10% better performance than the dummy
    estimator. Distinguish between [first, all, mean] fold(s).
    If not further testing of the configuration is skipped, as it is regarded as not promising enough.
    """

    def __init__(self,  metric: str='', smaller_than: float =1., strategy='first'):
        self.metric = metric
        self.smaller_than = smaller_than
        self.dummy_performance = None
        self.comparative_value = None
        try:
            self.strategy = enum_strategy[strategy]
        except:
            raise AttributeError("Your strategy: "+str(strategy)+" is not supported yet. Please use one of "+str([x.name for x in enum_strategy]))

    def set_dummy_performance(self, dummy_result):
        self.comparative_value = dummy_result.validation.metrics[self.metric]+self.smaller_than

    def shall_continue(self, config_item):
        if self.strategy.name == 'first':
            if config_item.inner_folds[0].validation.metrics[self.metric] < self.comparative_value:
                return False
        elif self.strategy.name == 'all':
            if all(item > self.comparative_value for item in
                   [x.validation.metrics[self.metric] for x in config_item.inner_folds]):
                return False
        elif self.strategy.name == 'mean':
            if np.mean([x.validation.metrics[self.metric] for x in config_item.inner_folds]) < self.comparative_value:
                return False
        return True

