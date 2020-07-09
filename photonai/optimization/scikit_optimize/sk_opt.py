from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.space import Categorical as skoptCategorical
from skopt.plots import plot_evaluations, plot_objective

import numpy as np
import matplotlib.pylab as plt

from photonai.optimization import FloatRange, IntegerRange
from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.optimization import Categorical as PhotonCategorical
from photonai.photonlogger.logger import logger


class SkOptOptimizer(PhotonSlaveOptimizer):

    def __init__(self, n_configurations: int=20, acq_func: str = 'gp_hedge', acq_func_kwargs: dict = None):
        self.optimizer = None
        self.hyperparameter_list = []
        self.metric_to_optimize = ''
        self.ask = self.ask_generator()
        self.n_configurations = n_configurations
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs
        self.maximize_metric = True
        self.constant_dictionary = {}

    def prepare(self, pipeline_elements: list, maximize_metric: bool):

        self.hyperparameter_list = []
        self.maximize_metric = maximize_metric
        # build space
        space = []
        for pipe_element in pipeline_elements:
            if hasattr(pipe_element, 'hyperparameters'):
                for name, value in pipe_element.hyperparameters.items():
                    # if we only have one value we do not need to optimize
                    if isinstance(value, list) and len(value) < 2:
                        self.constant_dictionary[name] = value[0]
                        continue
                    if isinstance(value, PhotonCategorical) and len(value.values) < 2:
                        self.constant_dictionary[name] = value.values[0]
                        continue
                    skopt_param = self._convert_PHOTON_to_skopt_space(value, name)
                    if skopt_param is not None:
                        space.append(skopt_param)
        if len(space) == 0:
            logger.warning("Did not find any hyperparameters to convert into skopt space")
            self.optimizer = None
        else:
            self.optimizer = Optimizer(space, "ET", acq_func=self.acq_func, acq_func_kwargs=self.acq_func_kwargs)
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
            if hyperparam.range_type == 'linspace':
                return Real(hyperparam.start, hyperparam.stop, name=name, prior='uniform')
            elif hyperparam.range_type == 'logspace':
                return Real(hyperparam.start, hyperparam.stop, name=name, prior='log-uniform')
            else:
                return Real(hyperparam.start, hyperparam.stop, name=name)
        elif isinstance(hyperparam, IntegerRange):
            return Integer(hyperparam.start, hyperparam.stop, name=name)

    def ask_generator(self):
        if self.optimizer is None:
            yield {}
        else:
            for i in range(self.n_configurations):
                next_config_list = self.optimizer.ask()
                next_config_dict = {self.hyperparameter_list[number]: self._convert_to_native(value) for number, value in enumerate(next_config_list)}
                yield next_config_dict

    def _convert_to_native(self, obj):
        # check if we have a numpy object, if so convert it to python native
        if type(obj).__module__ == np.__name__:
            return np.asscalar(obj)
        else:
            return obj
