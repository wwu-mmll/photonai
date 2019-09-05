
class PhotonBaseOptimizer:
    """
    The PHOTON interface for hyperparameter search optimization algorithms.
    """

    def __init__(self, *kwargs):
        pass

    def prepare(self, pipeline_elements: list, maximize_metric: bool):
        """
        Initializes hyperparameter search.
        Assembles all hyperparameters of the pipeline_element list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.
        """
        pass

    def ask(self):
        """
        When called, returns the next configuration that should be tested.

        Returns
        -------
        next config to test
        """
        pass

    def tell(self, config, performance):
        """
        Parameters
        ----------
        * 'config' [dict]:
            The configuration that has been trained and tested
        * 'performance' [dict]:
            Metrics about the configuration's generalization capabilities.
        """
        pass

    def plot(self, results_folder):
        """
        Plot optimizer specific visualizations
        :param results_folder:
        :return:
        """
        pass

    def plot_objective(self):
        """
        Uses plot_objective function of Scikit-Optimize to plot hyperparameters and partial dependences.
        :return:
        matplotlib figure
        """
        raise NotImplementedError('plot_objective is not yet available for this optimizer. Currently supported for'
                                  'skopt.')

    def plot_evaluations(self):
        """
        Uses plot_evaluations function of Scikit-Optimize to plot hyperparameters and respective performance estimates.
        :return:
        matplotlib figure
        """
        raise NotImplementedError('plot_evaluations is not yet available for this optimizer. Currently supported for'
                                  'skopt.')


