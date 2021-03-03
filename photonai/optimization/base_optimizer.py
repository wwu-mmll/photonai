from typing import Callable


class PhotonSlaveOptimizer(object):
    """Photon slave optimizer.

    Base class for optimizer in PHOTONAI.
    The PhotonSlaveOptimizer is controlled by PHOTONAIs OuterFoldManager.
    With the ask-tell principle PHOTONAI gets new configs.
    It terminates by some specific criteria, that leads to an empty ask generator.

    """
    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """
        Initializes hyperparameter search.

        Assembles all hyperparameters of the list of PipelineElements
        in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

        """
        pass

    def ask(self) -> dict:
        """
        When ask is called it returns the next configuration to be tested.

        Returns:
            Dict of configurations with parameters to be tested next.

        """
        pass

    def tell(self, config: dict, performance: float) -> None:
        """
        Returns the performance of a tested configuration to calculate new ones.

        Parameters:
            config: dict
                The configuration that has been trained and tested.

            performance: float
                Metrics about the configuration's generalization capabilities.

        """
        pass


class PhotonMasterOptimizer(object):
    """Photon master optimizer.

    Base class for optimizer in PHOTONAI.
    The PhotonMasterOptimizer controls PHOTONAIs optimization process.
    PHOTONAI creates an objective function that is used by the optimizer.
    The limitation of runs and configs of the
    objective function is up to PhotonMasterOptimizer.

    """
    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function: Callable) -> None:
        """
        Initializes hyperparameter search.

        Assembles all hyperparameters of the list of PipelineElements
        in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

            objective_function:
                The cost or objective function.

        """
        pass

    def optimize(self) -> None:
        """Start the optimization process based on the underlying objective function."""
        pass
