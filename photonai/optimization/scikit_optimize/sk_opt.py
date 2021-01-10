from skopt import Optimizer
from skopt.space import Real, Integer, Dimension
from skopt.space import Categorical as skoptCategorical

import warnings
import numpy as np
from typing import Union, Generator
import sklearn

from photonai.optimization import FloatRange, IntegerRange, BooleanSwitch, PhotonHyperparam
from photonai.optimization.base_optimizer import PhotonSlaveOptimizer
from photonai.optimization import Categorical as PhotonCategorical
from photonai.photonlogger.logger import logger


class SkOptOptimizer(PhotonSlaveOptimizer):
    """Wrapper for scikit-optimize with PHOTONAI.

    scikit-optimiz usage and implementation details:
    https://scikit-optimize.github.io/stable/

    A detailed parameter documentation:
    https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer

    Parameters
    ----------
    n_configurations: int, default=20
        Number of configurations to be calculated.

    n_initial_points: int, default=10
        Number of evaluations with initialization points
        before approximating it with `base_estimator`.

    base_estimator: str or sklearn.base.RegressorMixin, default='ET'
        Estimator for returning std(Y | x) along with E[Y | x].

    initial_point_generator: str, default='random'
        Generator for initial points.

    acq_func: dict, default=None (warning scenario with wallclock_limit = 60*5)
        Function to minimize over the posterior distribution. Can be either

    acq_func_kwargs: dict, default=None
        Additional arguments to be passed to the acquisition function.

    """
    def __init__(self,
                 n_configurations: int = 20,
                 n_initial_points: int = 10,
                 base_estimator: Union[str, sklearn.base.RegressorMixin] = "ET",
                 initial_point_generator: str = "random",
                 acq_func: str = 'gp_hedge',
                 acq_func_kwargs: dict = None):
        self.metric_to_optimize = ''
        self.n_configurations = n_configurations
        self.n_initial_points = n_initial_points
        self.base_estimator = base_estimator
        self.initial_point_generator = initial_point_generator
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        self.optimizer = None
        self.maximize_metric = None
        self.hyperparameter_list = []
        self.constant_dictionary = {}
        self.ask = self.ask_generator()

    def ask_generator(self) -> Generator:
        if self.optimizer is None:
            yield {}
        else:
            for i in range(self.n_configurations):
                next_config_list = self.optimizer.ask()
                next_config_dict = {self.hyperparameter_list[number]:
                                    self._convert_to_native(value) for number, value in enumerate(next_config_list)}
                yield next_config_dict

    def prepare(self, pipeline_elements: list, maximize_metric: bool) -> None:
        """Initializes hyperparameter search with scikit-optimize.

        Assembles all hyperparameters of the pipeline_element
        list in order to prepare the hyperparameter search space.
        Hyperparameters can be accessed via pipe_element.hyperparameters.

        Parameters
        ----------
        pipeline_elements: list
            List of all pipeline_elements to create hyperparameter_space.

        maximize_metric: bool
            Boolean for distinguish between score and error.

        """
        self.optimizer = None
        self.hyperparameter_list = []
        self.maximize_metric = maximize_metric

        # build skopt space
        space = []
        for pipe_element in pipeline_elements:
            if pipe_element.__class__.__name__ == 'Switch':
                error_msg = 'Scikit-Optimize cannot operate in the specified hyperparameter space with a Switch ' \
                            'element. We recommend the use of SMAC.'
                logger.error(error_msg)
                raise ValueError(error_msg)

            if hasattr(pipe_element, 'hyperparameters'):
                for name, value in pipe_element.hyperparameters.items():
                    # if we only have one value we do not need to optimize
                    if isinstance(value, list) and len(value) < 2:
                        self.constant_dictionary[name] = value[0]
                        continue
                    if isinstance(value, PhotonCategorical) and len(value.values) < 2:
                        self.constant_dictionary[name] = value.values[0]
                        continue
                    skopt_param = self._convert_photonai_to_skopt_space(value, name)
                    if skopt_param is not None:
                        space.append(skopt_param)

        if self.constant_dictionary:
            msg = "PHOTONAI has detected some one-valued params in your hyperparameters. Pleas use the kwargs for " \
                  "constant values. This run ignores following settings: " + str(self.constant_dictionary.keys())
            logger.warning(msg)
            warnings.warn(msg)

        if len(space) == 0:
            msg = "Did not find any hyperparameter to convert into skopt space."
            logger.warning(msg)
            warnings.warn(msg)
        else:
            self.optimizer = Optimizer(space,
                                       base_estimator=self.base_estimator,
                                       n_initial_points=self.n_initial_points,
                                       initial_point_generator=self.initial_point_generator,
                                       acq_func=self.acq_func,
                                       acq_func_kwargs=self.acq_func_kwargs)
        self.ask = self.ask_generator()

    def tell(self, config: dict, performance: float) -> None:
        """Provide result for skopt optimizer to calculate new ones.

        Parameters
         ----------
        config: dict
            The configuration that has been trained and tested.

        performance: dict
            Metrics about the configuration's generalization capabilities.

        """
        # convert dictionary to list in correct order
        if self.optimizer is not None:
            config_values = [config[name] for name in self.hyperparameter_list]
            best_config_metric_performance = performance
            if self.maximize_metric:
                best_config_metric_performance = -best_config_metric_performance
            self.optimizer.tell(config_values, best_config_metric_performance)

    def _convert_photonai_to_skopt_space(self, hyperparam: Union[PhotonHyperparam, list], name: str) -> Dimension:
        self.hyperparameter_list.append(name)
        if isinstance(hyperparam, PhotonCategorical) or isinstance(hyperparam, BooleanSwitch):
            return skoptCategorical(hyperparam.values, name=name)
        elif isinstance(hyperparam, list):
            return skoptCategorical(hyperparam, name=name)
        elif isinstance(hyperparam, FloatRange):
            if hyperparam.range_type == 'linspace':
                return Real(hyperparam.start, hyperparam.stop, name=name, prior='uniform')
            elif hyperparam.range_type == 'logspace':
                return Real(hyperparam.start, hyperparam.stop, name=name, prior='log-uniform')
            else:
                msg = "The hyperparam.range_type "+hyperparam.range_type+" is not supported by scikit-optimize."
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(hyperparam, IntegerRange):
            return Integer(hyperparam.start, hyperparam.stop, name=name)

        msg = "Cannot convert hyperparameter " + str(hyperparam) + ". " \
              "Supported types: Categorical, IntegerRange, FloatRange, list."
        logger.error(msg)
        raise ValueError(msg)

    @staticmethod
    def _convert_to_native(obj):
        # check if we have a numpy object, if so convert it to python native
        if type(obj).__module__ == np.__name__:
            return obj.item()
        else:
            return obj
