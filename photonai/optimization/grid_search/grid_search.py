import datetime

import numpy as np

from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.optimization.config_grid import create_global_config_grid
from photonai.photonlogger.logger import logger


class GridSearchOptimizer(PhotonSlaveOptimizer):
    """
    Searches for the best configuration by iteratively testing all possible hyperparameter combinations.
    """
    def __init__(self):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.ask = self.next_config_generator()

    def prepare(self, pipeline_elements, maximize_metric):
        self.pipeline_elements = pipeline_elements
        self.ask = self.next_config_generator()
        self.param_grid = create_global_config_grid(self.pipeline_elements)
        logger.info("Grid Search generated " + str(len(self.param_grid)) + " configurations")

    def next_config_generator(self):
        for parameters in self.param_grid:
            yield parameters

    def tell(self, config, performance):
        # influence return value of next_config
        pass


class RandomGridSearchOptimizer(GridSearchOptimizer):
    """
     Searches for the best configuration by randomly testing k possible hyperparameter combinations.
    """

    def __init__(self, n_configurations=25):
        super(RandomGridSearchOptimizer, self).__init__()
        self._k = n_configurations
        self.n_configurations = self._k

    def prepare(self, pipeline_elements, maximize_metric):
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements, maximize_metric)
        self.n_configurations = self._k
        self.param_grid = list(self.param_grid)
        # create random chaos in list
        np.random.shuffle(self.param_grid)
        if self.n_configurations is not None:
            # k is maximal all grid items
            if self.n_configurations > len(self.param_grid):
                self.n_configurations = len(self.param_grid)
            self.param_grid = self.param_grid[0:self.n_configurations]


class TimeBoxedRandomGridSearchOptimizer(RandomGridSearchOptimizer):
    """
    Iteratively tests k possible hyperparameter configurations until a certain time limit is reached.
    """

    def __init__(self, limit_in_minutes=60, n_configurations=None):
        super(TimeBoxedRandomGridSearchOptimizer, self).__init__(n_configurations)
        self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

    def prepare(self, pipeline_elements, maximize_metric):
        super(TimeBoxedRandomGridSearchOptimizer, self).prepare(pipeline_elements, maximize_metric)
        self.start_time = None

    def next_config_generator(self):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(TimeBoxedRandomGridSearchOptimizer, self).next_config_generator():
            if datetime.datetime.now() < self.end_time:
                yield parameters
