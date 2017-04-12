
from itertools import product

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
            # if any(isinstance(el, list) for el in p_element.config_grid):
            #     for item in p_element.config_grid:
            #         possible_configurations.append(item)
            # else:
            possible_configurations.append(p_element.config_grid)
        self.param_grid = product(*possible_configurations)
        # Todo: _check_param_grid(param_grid)
        # self.parameter_iterable = ParameterGrid(self.param_grid)
        self.parameter_iterable = self.param_grid

    def next_config_generator(self):
        for parameters in self.parameter_iterable:
            param_dict = {}
            for item in parameters:
                param_dict.update(item)
            yield param_dict

    def evaluate_recent_performance(self, config, performance):
        # influence return value of next_config
        pass


class AnyHyperparamOptimizer(object):
    def __init__(self, params_to_optimize):
        self.params_to_optimize = params_to_optimize
        self.next_config = self.next_config_generator()
        self.next_config_to_try = 1

    def next_config_generator(self):
        yield self.next_config_to_try

    def evaluate_recent_performance(self, config, performance):
        # according to the last performance for the given config,
        # the next item should be chosen wisely
        self.next_config_to_try = self.params_to_optimize(2)
