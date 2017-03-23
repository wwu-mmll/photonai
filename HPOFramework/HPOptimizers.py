
from sklearn.model_selection._search import ParameterGrid


class GridSearchOptimizer(object):
    def __init__(self, param_grid):
        self.param_grid = param_grid
        # Todo: _check_param_grid(param_grid)
        self.parameter_iterable = ParameterGrid(self.param_grid)
        self.next_config = self.next_config_generator()

    def next_config_generator(self):
        for parameters in self.parameter_iterable:
            yield parameters

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
