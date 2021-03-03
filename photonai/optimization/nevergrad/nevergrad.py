from typing import Callable
import warnings

from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange, BooleanSwitch, PhotonHyperparam
from photonai.optimization.base_optimizer import PhotonMasterOptimizer
from photonai.photonlogger import logger

try:
    import nevergrad as ng
    from nevergrad.optimization.base import Optimizer
    __found__ = True
except ModuleNotFoundError:
    __found__ = False


class NevergradOptimizer(PhotonMasterOptimizer):
    """Nevergrad Wrapper for PHOTONAI.

    Nevergrad is a gradient-free optimization platform.

    Nevergrad [usage and implementation details](
    https://facebookresearch.github.io/nevergrad/).

    Example:
        ``` python
        import nevergrad as ng
        # list of all available nevergrad optimizer
        print(list(ng.optimizers.registry.values()))

        my_pipe = Hyperpipe('nevergrad_example',
                            optimizer='nevergrad',
                            optimizer_params={'facade': 'NGO', 'n_configurations': 30},
                            ...
                            )
        ```

    """
    def __init__(self, facade='NGO', n_configurations: int = 100, rng: int = 42):
        """
        Initialize the object.

        Parameters:
            facade:
                Choice of the Nevergrad backend strategy, e.g. [NGO, ...].

            n_configurations:
                Number of runs.

            rng:
                Random Seed.

        """

        if not __found__:
            msg = "Module nevergrad not found or not installed as expected. " \
                  "Please install the nevergrad/requirements.txt PHOTONAI provides."
            logger.error(msg)
            raise ModuleNotFoundError(msg)

        if facade in list(ng.optimizers.registry.values()):
            self.facade = facade
        elif facade in list(ng.optimizers.registry.keys()):
            self.facade = ng.optimizers.registry[facade]
        else:
            msg = "nevergrad.optimizer {} not known. Check out all available nevergrad optimizers " \
                  "by nevergrad.optimizers.registry.keys()".format(str(facade))
            logger.error(msg.format(str(facade)))
            raise ValueError(msg.format(str(facade)))

        self.n_configurations = n_configurations
        self.space = None  # Hyperparameter space for nevergrad
        self.switch_optiones = {}
        self.hyperparameters = []
        self.rng = rng

        self.maximize_metric = False
        self.constant_dictionary = {}

        self.objective = None
        self.optimizer = None

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function: Callable) -> None:
        """Prepare Nevergrad Optimizer.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

            objective_function:
                The cost or objective function.

        """
        self.space = self._build_nevergrad_space(pipeline_elements)
        self.space.random_state.seed(self.rng)
        if self.constant_dictionary:
            msg = "PHOTONAI has detected some one-valued params in your hyperparameters. Pleas use the kwargs for " \
                  "constant values. This run ignores following settings: " + str(self.constant_dictionary.keys())
            logger.warning(msg)
            warnings.warn(msg)
        self.maximize_metric = maximize_metric

        def nevergrad_objective_function(**current_config):
            return objective_function(current_config)
        self.objective = nevergrad_objective_function

        self.optimizer = self.facade(parametrization=self.space, budget=self.n_configurations)

    def optimize(self) -> None:
        self.optimizer.minimize(self.objective)

    def _build_nevergrad_space(self, pipeline_elements: list):
        """
        Build entire Nevergrad hyperparameter space.

        Parameters:
            pipeline_elements:
                List of all pipeline_elements to create the hyperparameter space.

        """
        param_dict = {}
        for pipe_element in pipeline_elements:
            # build conditions for switch elements
            if pipe_element.__class__.__name__ == 'Switch':
                raise NotImplementedError("Currently PHOTONAIs Switch is not supported by nevergrad.")

            if hasattr(pipe_element, 'hyperparameters'):
                for name, value in pipe_element.hyperparameters.items():
                    self.hyperparameters.append(name)
                    # if we only have one value we do not need to optimize
                    if isinstance(value, list) and len(value) < 2:
                        self.constant_dictionary[name] = value[0]
                        continue
                    if isinstance(value, PhotonCategorical) and len(value.values) < 2:
                        self.constant_dictionary[name] = value.values[0]
                        continue
                    nevergrad_param = self._convert_photonai_to_nevergrad_param(value)
                    if nevergrad_param is not None:
                        param_dict[name] = nevergrad_param
        return ng.p.Instrumentation(**param_dict)

    @staticmethod
    def _convert_photonai_to_nevergrad_param(hyperparam: PhotonHyperparam):
        """
        Helper function: Convert PHOTONAI to Nevergrad hyperparameter.

        Parameters:
            hyperparam:
                 One of photonai.optimization.hyperparameters.

        """
        if isinstance(hyperparam, PhotonCategorical) or isinstance(hyperparam, BooleanSwitch):
            return ng.p.Choice(hyperparam.values)
        elif isinstance(hyperparam, list):
            return ng.p.Choice(hyperparam)
        elif isinstance(hyperparam, FloatRange):
            if hyperparam.range_type == 'linspace':
                return ng.p.Scalar(lower=hyperparam.start, upper=hyperparam.stop)
            elif hyperparam.range_type == 'logspace':
                return ng.p.Log(lower=hyperparam.start, upper=hyperparam.stop)
            msg = str(hyperparam.range_type) + "in your float hyperparameter is not implemented yet."
            logger.error(msg)
            raise NotImplementedError(msg)
        elif isinstance(hyperparam, IntegerRange):
            return ng.p.Scalar(lower=hyperparam.start, upper=hyperparam.stop).set_integer_casting()

        msg = "Cannot convert hyperparameter " + str(hyperparam) + ". Supported types: Categorical, IntegerRange," \
                                                                   "FloatRange, list."
        logger.error(msg)
        raise ValueError(msg)
