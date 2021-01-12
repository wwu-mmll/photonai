import datetime

import numpy as np
from typing import Union, Generator

from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.optimization.config_grid import create_global_config_grid
from photonai.photonlogger.logger import logger


class GridSearchOptimizer(PhotonSlaveOptimizer):
    """Grid search optimizer.

    Searches for the best configuration by iteratively
    testing all possible hyperparameter combinations.

    """
    def __init__(self):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.ask = self.next_config_generator()

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """Initializes grid based hyperparameter search.

        Creates a grid from input pipeline_elements.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        pipeline_elements: list
            List of all pipeline_elements to create hyperparameter_space.

        maximize_metric: bool
            Boolean for distinguish between score and error.

        """
        self.pipeline_elements = pipeline_elements
        self.ask = self.next_config_generator()
        self.param_grid = create_global_config_grid(self.pipeline_elements)
        logger.info("Grid Search generated " + str(len(self.param_grid)) + " configurations")

    def next_config_generator(self) -> Generator:
        for parameters in self.param_grid:
            yield parameters


class RandomGridSearchOptimizer(GridSearchOptimizer):
    """Random grid search optimizer.

    Searches for the best configuration by randomly
    testing n possible hyperparameter combinations.

    Parameters
    ----------
    n_configurations: int or None, default=25
        Number of configurations to be calculated.

    """
    def __init__(self, n_configurations: Union[int, None] = 25):
        super(RandomGridSearchOptimizer, self).__init__()
        self._k = n_configurations
        self.n_configurations = self._k

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """Initializes hyperparameter search.

        Parameters
        ----------
        pipeline_elements: list
            List of all pipeline_elements to create hyperparameter_space.

        maximize_metric: bool
            Boolean for distinguish between score and error.

        """
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements, maximize_metric)
        self.n_configurations = self._k
        self.param_grid = list(self.param_grid)
        # create random order in list
        np.random.shuffle(self.param_grid)
        if self.n_configurations is not None:
            # k is maximal all grid items
            if self.n_configurations > len(self.param_grid):
                self.n_configurations = len(self.param_grid)
            self.param_grid = self.param_grid[0:self.n_configurations]


class TimeBoxedRandomGridSearchOptimizer(RandomGridSearchOptimizer):
    """Time boxed random grid serach.

    Iteratively tests n possible hyperparameter configurations
    until a certain time limit is reached.

    Parameters
    ----------
    limit_in_minutes: int, default=60
        Total time in minutes.

    n_configurations: int or None, default=25
        Number of configurations to be calculated.

    """
    def __init__(self, limit_in_minutes: float = 60, n_configurations: Union[int, None] = None):
        super(TimeBoxedRandomGridSearchOptimizer, self).__init__(n_configurations)
        self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """Initializes hyperparameter search.

        Parameters
        ----------
        pipeline_elements: list
            List of all pipeline_elements to create hyperparameter_space.

        maximize_metric: bool
            Boolean for distinguish between score and error.

        """
        super(TimeBoxedRandomGridSearchOptimizer, self).prepare(pipeline_elements, maximize_metric)
        self.start_time = None

    def next_config_generator(self) -> Generator:
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(TimeBoxedRandomGridSearchOptimizer, self).next_config_generator():
            if datetime.datetime.now() < self.end_time:
                yield parameters
