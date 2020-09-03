
class PhotonBaseOptimizer:
    """
    The PHOTON interface for hyperparameter search optimization algorithms.
    """

    def __init__(self, *kwargs):
        pass


class PhotonSlaveOptimizer(PhotonBaseOptimizer):
    """
    The PhotonSlaveOptimizer is controlled by PHOTON. By ask-tell principle PHOTON get new configs.
    It terminates by some specific criteria with an aks -> empty yield.
    """

    def __init__(self, *kwargs):
        super(PhotonSlaveOptimizer, self).__init__(kwargs)

    def prepare(self, pipeline_elements: list, maximize_metric: bool):
        """
        Initializes hyperparameter search.
        Assembles all hyperparameters of the pipeline_element list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        * `maximize_metric` [bool]:
            Boolean for distinguish between score and error.
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


class PhotonMasterOptimizer(PhotonBaseOptimizer):
    """
        The PhotonMasterOptimizer controls PHOTON. PHOTON provides an objective_function.
        The runs and configs of the objective_function is up to PhotonMasterOptimizer.
        """

    def __init__(self, *kwargs):
        super(PhotonMasterOptimizer, self).__init__(kwargs)

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function):
        """
        Initializes hyperparameter search.
        Assembles all hyperparameters of the pipeline_element list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        * `maximize_metric` [bool]:
            Boolean for distinguish between score and error.
        * `objective_function` [callable]:
            The cost or objective function.
        """
        pass

    def optimize(self):
        """
        Start optimization over objective_function.
        """
        pass
