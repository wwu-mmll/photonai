
from itertools import product
import numpy as np
import datetime


class GridSearchOptimizer(object):
    def __init__(self):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.next_config = self.next_config_generator()

    def prepare(self, pipeline_elements):
        self.pipeline_elements = pipeline_elements
        possible_configurations = []
        for p_element in self.pipeline_elements:
            possible_configurations.append(p_element.config_grid)
        self.param_grid = product(*possible_configurations)

    def next_config_generator(self):
        for parameters in self.param_grid:
            param_dict = {}
            for item in parameters:
                param_dict.update(item)
            yield param_dict

    def evaluate_recent_performance(self, config, performance):
        # influence return value of next_config
        pass

class RandomGridSearchOptimizer(GridSearchOptimizer):

    def __init__(self, k=None):
        super(RandomGridSearchOptimizer, self).__init__()
        self.k = k

    def prepare(self, pipeline_elements):
        self.pipeline_elements = pipeline_elements
        possible_configurations = []
        for p_element in self.pipeline_elements:
            possible_configurations.append(p_element.config_grid)
        self.param_grid = list(product(*possible_configurations))
        # create random chaos in list
        np.random.shuffle(self.param_grid)
        if self.k is not None:
            self.param_grid = self.param_grid[0:self.k]


class TimeBoxedRandomGridSearchOptimizer(RandomGridSearchOptimizer):

    def __init__(self, limit_in_minutes=60):
        super(TimeBoxedRandomGridSearchOptimizer, self).__init__()
        self.limit_in_minutes = limit_in_minutes
        self.start_time = None
        self.end_time = None

    def next_config_generator(self):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            self.end_time = self.start_time + datetime.timedelta(minutes=self.limit_in_minutes)
        for parameters in super(TimeBoxedRandomGridSearchOptimizer, self).next_config_generator():
            if datetime.datetime.now() < self.end_time:
                yield parameters


# class AnyHyperparamOptimizer(object):
#     def __init__(self, params_to_optimize):
#         self.params_to_optimize = params_to_optimize
#         self.next_config = self.next_config_generator()
#         self.next_config_to_try = 1
#
#     def next_config_generator(self):
#         yield self.next_config_to_try
#
#     def evaluate_recent_performance(self, config, performance):
#         # according to the last performance for the given config,
#         # the next item should be chosen wisely
#         self.next_config_to_try = self.params_to_optimize(2)
