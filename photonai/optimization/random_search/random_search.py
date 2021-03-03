import datetime
import random
from typing import Union, Generator

from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.photonlogger.logger import logger


class RandomSearchOptimizer(PhotonSlaveOptimizer):
    """Random search optimizer.

    Searches for the best configuration by randomly
    testing hyperparameter combinations without any grid.

    """
    def __init__(self, limit_in_minutes: Union[float, None] = 60, n_configurations: Union[int, None] = None):
        """
        Initialize the object.
        One of limit_in_minutes or n_configurations must differ from None.

        Parameters:
            limit_in_minutes:
                Total time in minutes.

            n_configurations:
                Number of configurations to be calculated.

        """
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.ask = self.next_config_generator()
        self.n_configurations = None

        if not limit_in_minutes or limit_in_minutes <= 0:
            self.limit_in_minutes = None
        else:
            self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

        if not n_configurations or n_configurations <= 0:
            self.n_configurations = None
        else:
            self.n_configurations = n_configurations
        self.k_configutration = 0  # use k++ until k==n: break

        if self.n_configurations is None and self.limit_in_minutes is None:
            msg = "No stopping criteria for RandomSearchOptimizer."
            logger.error(msg)
            raise ValueError(msg)

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """
        Initializes grid free random hyperparameter search.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

        """
        self.start_time = None
        self.pipeline_elements = pipeline_elements
        self.ask = self.next_config_generator()

    def next_config_generator(self) -> Generator:
        """
        Generator for new configs - ask method.

        Returns:
            Yields the next config.

        """
        while True:
            _ = (yield self._generate_config())
            self.k_configutration += 1
            if self.limit_in_minutes:
                if self.start_time is None:
                    self.start_time = datetime.datetime.now()
                    self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)

                if datetime.datetime.now() >= self.end_time:
                    return

            if self.n_configurations:
                if self.k_configutration >= self.n_configurations:
                    return

    def _generate_config(self):
        config = {}
        for p_element in self.pipeline_elements:
            for h_key, h_value in p_element.hyperparameters.items():
                if isinstance(h_value, list):
                    config[h_key] = random.choice(h_value)
                else:
                    config[h_key] = h_value.get_random_value()
        return config
