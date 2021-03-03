import datetime

import numpy as np
from typing import Union, Generator

from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.optimization.config_grid import create_global_config_grid
from photonai.photonlogger.logger import logger


class GridSearchOptimizer(PhotonSlaveOptimizer):
    """Grid search optimizer.

    Searches for the best configuration by iteratively
    testing a grid of possible hyperparameter combinations.

    Example:
        ``` python
        my_pipe = Hyperpipe(name='grid_based_pipe',
                            optimizer='grid_search',
                            ...
                            )
        my_pipe.fit(X, y)
        ```

    """
    def __init__(self):
        """Initialize the object."""
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.ask = self.next_config_generator()

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """
        Creates a grid from a list of PipelineElements.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

        """
        self.pipeline_elements = pipeline_elements
        self.ask = self.next_config_generator()
        self.param_grid = create_global_config_grid(self.pipeline_elements)
        logger.info("Grid Search generated " + str(len(self.param_grid)) + " configurations")

    def next_config_generator(self) -> Generator:
        """
        Generator for new configs - ask method.

        Returns:
            Yields the next config.

        """
        for parameters in self.param_grid:
            yield parameters


class RandomGridSearchOptimizer(GridSearchOptimizer):
    """Random grid search optimizer.

    Searches for the best configuration by randomly
    testing n points of a grid of possible hyperparameters.

    Example:
        ``` python
        my_pipe = Hyperpipe(name='rgrid_based_pipe',
                            optimizer='random_grid_search',
                            optimizer_params={'n_configurations': 50,
                                              'limit_in_minutes': 10},
                            ...
                            )
        my_pipe.fit(X, y)
        ```

    """
    def __init__(self, limit_in_minutes: Union[float, None] = None, n_configurations: Union[int, None] = 25):
        """
        Initialize the object.

        Parameters:
            limit_in_minutes:
                Total time in minutes.

            n_configurations:
                Number of configurations to be calculated.

        """
        super(RandomGridSearchOptimizer, self).__init__()
        self._k = n_configurations
        self.n_configurations = self._k
        self.limit_in_minutes = limit_in_minutes
        self.start_time, self.end_time = None, None

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """
        Prepare hyperparameter search.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

        """
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements, maximize_metric)
        self.start_time = None
        self.n_configurations = self._k
        self.param_grid = list(self.param_grid)
        # create random order in list
        np.random.shuffle(self.param_grid)
        if self.n_configurations is not None:
            # k is maximal all grid items
            if self.n_configurations > len(self.param_grid):
                self.n_configurations = len(self.param_grid)
            self.param_grid = self.param_grid[0:self.n_configurations]

    def next_config_generator(self) -> Generator:
        """
        Generator for new configs - ask method.

        Returns:
            Yields the next config.

        """
        if self.start_time is None and self.limit_in_minutes is not None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(RandomGridSearchOptimizer, self).next_config_generator():
            if self.limit_in_minutes is None or datetime.datetime.now() < self.end_time:
                yield parameters
