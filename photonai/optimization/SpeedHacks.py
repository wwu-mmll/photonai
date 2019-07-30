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


class MinimumPerformance:
    """
    Tests if a configuration performs better than a given limit for a particular metric.

    Example
    -------
    MinimumPerformance('accuracy', 0.96) tests if the configuration has at least a performance of 0.96 in the first fold.
    If not further testing of the configuration is skipped, as it is regarded as not promising enough.
    """

    def __init__(self,  metric, smaller_than):
        self.metric = metric
        self.smaller_than = smaller_than

    def shall_continue(self, config_item):
        if config_item.inner_folds[0].validation.metrics[self.metric] < self.smaller_than:
            return False
        else:
            return True

