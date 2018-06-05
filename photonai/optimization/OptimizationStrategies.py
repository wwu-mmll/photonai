import datetime
import numpy as np
from itertools import product
from .Hyperparameters import FloatRange, Categorical, IntegerRange, BooleanSwitch, PhotonHyperparam
from sklearn.model_selection import ParameterGrid
from itertools import product

class GridSearchOptimizer(object):
    def __init__(self):
        self.param_grid = []
        self.pipeline_elements = None
        self.parameter_iterable = None
        self.next_config = self.next_config_generator()

    def prepare(self, pipeline_elements):
        self.pipeline_elements = pipeline_elements
        self.next_config = self.next_config_generator()
        possible_configurations = []

        global_hyperparameter_dict = {}

        for p_element in self.pipeline_elements:
            if len(p_element.sklearn_hyperparams) > 0:
                for h_key, h_value in p_element.sklearn_hyperparams.items():
                    if isinstance(h_value, list):
                        global_hyperparameter_dict[h_key] = h_value
                    elif isinstance(h_value, PhotonHyperparam):
                        # when we have a range we need to convert it to a definite list of values
                        if isinstance(h_value, FloatRange) or isinstance(h_value, IntegerRange):
                            # build a definite list of values
                            h_value.transform()
                            global_hyperparameter_dict[h_key] = h_value.values
                        elif isinstance(h_value, BooleanSwitch) or isinstance(h_value, Categorical):
                            global_hyperparameter_dict[h_key] = h_value.values
            #
            # if p_element.config_grid:
            #     possible_configurations.append(p_element.config_grid)
        self.param_grid = list(ParameterGrid(global_hyperparameter_dict))
        # generate Parameter Grid
        # if len(possible_configurations) == 1:
        #     self.param_grid = [[i] for i in possible_configurations[0]]
        # else:
        #     self.param_grid = product(*possible_configurations)

    def next_config_generator(self):
        for parameters in self.param_grid:
            yield parameters

    def evaluate_recent_performance(self, config, performance):
        # influence return value of next_config
        pass


class RandomGridSearchOptimizer(GridSearchOptimizer):

    def __init__(self, k=None):
        super(RandomGridSearchOptimizer, self).__init__()
        self.k = k

    def prepare(self, pipeline_elements):
        super(RandomGridSearchOptimizer, self).prepare(pipeline_elements)
        self.param_grid = list(self.param_grid)
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

    def prepare(self, pipeline_elements):
        super(TimeBoxedRandomGridSearchOptimizer, self).prepare(pipeline_elements)
        self.start_time = None

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
