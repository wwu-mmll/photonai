from typing import Callable


class PhotonSlaveOptimizer(object):
    """PhotonSlaveOptimizer

    The PhotonSlaveOptimizer is controlled by PHOTONAI.
    With the ask-tell principle PHOTONAI gets new configs.
    It terminates by some specific criteria leads to empty yield ask.

    """

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """Initializes hyperparameter search.

        Assembles all hyperparameters of the pipeline_element
        list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        * `maximize_metric` [bool]:
            Boolean for distinguish between score and error.
        """
        pass

    def ask(self) -> dict:
        """
        When ask is called it returns the next configuration to be tested.

        Returns
        -------
        * _ [dict]:
            config_dict, the next config to be tested
        """
        pass

    def tell(self, config: dict, performance: float) -> None:
        """
        Provide result for optimizer to calculate new ones.

        Parameters
        ----------
        * 'config' [dict]:
            The configuration that has been trained and tested.
        * 'performance' [dict]:
            Metrics about the configuration's generalization capabilities.
        """
        pass


class PhotonMasterOptimizer(object):
    """PhotonMasterOptimizer

    The PhotonMasterOptimizer controls PHOTONAI.
    PHOTONAI creates an objective function that is used by the optimizer.
    The limitation of runs and configs of the
    objective function is up to PhotonMasterOptimizer.

    """

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function: Callable) -> None:
        """Initializes hyperparameter search.

        Assembles all hyperparameters of the pipeline_element
        list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        * `maximize_metric` [bool]:
            Boolean for distinguish between score and error.
        * `objective_function` [Callable]:
            The cost or objective function.
        """
        pass

    def optimize(self) -> None:
        """
        Start optimization over objective_function.
        """
        pass
