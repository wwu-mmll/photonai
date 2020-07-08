import datetime
import random

from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.photonlogger.logger import logger


class RandomSearchOptimizer(PhotonSlaveOptimizer):
    """
     Searches for the best configuration by randomly testing k possible hyperparameter combinations without grid.
    """

    def __init__(self, n_configurations=None, limit_in_minutes=60):
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

        if not n_configurations and limit_in_minutes <= 0:
            msg = "No stopping criteria for RandomSearchOptimizer."
            logger.warning(msg)

    def prepare(self, pipeline_elements, maximize_metric):
        self.pipeline_elements = pipeline_elements
        self.ask = self.next_config_generator()

    def next_config_generator(self):

        while True:
            val = (yield self.generate_config())
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

    def tell(self, config, performance):
        # influence return value of next_config
        pass

    def generate_config(self):
        config = {}
        for p_element in self.pipeline_elements:
            for h_key, h_value in p_element.hyperparameters.items():
                if isinstance(h_value, list):
                    config[h_key] = random.choice(h_value)
                else:
                    config[h_key] = h_value.get_random_value()
        return config



