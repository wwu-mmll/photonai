from .OptimizationStrategies import PhotonBaseOptimizer
from .Hyperparameters import FloatRange, IntegerRange
from .Hyperparameters import Categorical as PhotonCategorical
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.space import Categorical as skoptCategorical
import numpy as np


class SkOptOptimizer(PhotonBaseOptimizer):

    def __init__(self, num_iterations: int=20):
        self.optimizer = None
        self.hyperparameter_list = []
        self.metric_to_optimize = ''
        self.ask = self.ask_generator()
        self.num_iterations = num_iterations
        self.maximize_metric = True

    def prepare(self, pipeline_elements: list, maximize_metric: bool):

        self.hyperparameter_list = []
        self.maximize_metric = maximize_metric
        # build space
        space = []
        for pipe_element in pipeline_elements:
            for name, value in pipe_element.hyperparameters.items():
                skopt_param = self._convert_PHOTON_to_skopt_space(value, name)
                if skopt_param is not None:
                    space.append(skopt_param)
        self.optimizer = Optimizer(space, "ET")
        self.ask = self.ask_generator()

    def _convert_PHOTON_to_skopt_space(self, hyperparam: object, name: str):
        if not hyperparam:
            return None
        self.hyperparameter_list.append(name)
        if isinstance(hyperparam, PhotonCategorical):
            return skoptCategorical(hyperparam.values, name=name)
        elif isinstance(hyperparam, list):
            return skoptCategorical(hyperparam, name=name)
        elif isinstance(hyperparam, FloatRange):
            return Real(hyperparam.start, hyperparam.stop, name=name)
        elif isinstance(hyperparam, IntegerRange):
            return Integer(hyperparam.start, hyperparam.stop, name=name)

    def ask_generator(self):
        for i in range(self.num_iterations):
            next_config_list = self.optimizer.ask()
            next_config_dict = {self.hyperparameter_list[number]: self._convert_to_native(value) for number, value in enumerate(next_config_list)}
            yield next_config_dict

    def _convert_to_native(self, obj):
        # check if we have a numpy object, if so convert it to python native
        if type(obj).__module__ == np.__name__:
            return np.asscalar(obj)
        else:
            return obj

    def tell(self, config, performance):
        # convert dictionary to list in correct order
        config_values = [config[name] for name in self.hyperparameter_list]
        best_config_metric_performance = performance[1]
        if self.maximize_metric:
            best_config_metric_performance = -best_config_metric_performance
        # random_accuracy = np.random.randn(1)[0]
        self.optimizer.tell(config_values, best_config_metric_performance)
