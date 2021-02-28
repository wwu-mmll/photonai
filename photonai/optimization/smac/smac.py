import warnings
from typing import Callable

from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange, BooleanSwitch, PhotonHyperparam
from photonai.optimization.base_optimizer import PhotonMasterOptimizer
from photonai.photonlogger import logger

try:
    from smac.configspace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
        ConfigurationSpace, Configuration, InCondition, Constant
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.facade.smac_ac_facade import SMAC4AC
    from smac.facade.smac_bohb_facade import BOHB4HPO
    __found__ = True
except (ModuleNotFoundError, ImportError):
    __found__ = False


class SMACOptimizer(PhotonMasterOptimizer):
    """SMAC Wrapper for PHOTONAI.

    SMAC (sequential model-based algorithm configuration) is a
    versatile tool for optimizing algorithm parameters.
    The main core consists of Bayesian Optimization in
    combination with an aggressive racing mechanism to efficiently
    decide which of two configurations performs better.

    SMAC usage and implementation details [here](
    https://automl.github.io/SMAC3/master/quickstart.html).

    References:

        Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
        Sequential Model-Based Optimization for General Algorithm Configuration
        In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)

    Example:
        ``` python
        my_pipe = Hyperpipe('smac_example',
                            optimizer='smac',
                            optimizer_params={"facade": "SMAC4BO",
                                              "wallclock_limit": 60.0*10,  # seconds
                                              "ta_run_limit": 100},  # limit of configurations
                            ...)

        ```

    """
    def __init__(self, facade='SMAC4HPO',
                 run_obj: str = "quality",
                 deterministic: str = "true",
                 wallclock_limit: float = 60.0,
                 intensifier_kwargs: dict = None,
                 rng: int = 42, **kwargs):
        """
        Initialize the object.

        Parameters:
            facade:
                Choice of the SMAC backend strategy, [SMAC4BO, SMAC4HPO, SMAC4AC, BOHB4HPO].

            run_obj:
                Defines the optimization metric.
                When optimizing runtime, cutoff_time is required as well.

            wallclock_limit:
                Maximum amount of wallclock-time used for optimization.

            deterministic:
                If true, SMAC assumes that the target function or algorithm
                is deterministic (the same static seed of 0
                is always passed to the function/algorithm).
                If false, different random seeds are passed
                to the target function/algorithm.

            intensifier_kwargs:
                Dict for intensifier settings.

            rng:
                Random seed of SMAC.facade.

            **kwargs:
                All initial kwargs are passed to SMACs scenario.
                [List of all a vailable parameters](
                https://automl.github.io/SMAC3/master/options.html#scenario).

        """
        super(SMACOptimizer, self).__init__()

        if not __found__:
            msg = "Module smac not found or not installed as expected. " \
                  "Please install the smac/requirements.txt PHOTONAI provides."
            logger.error(msg)
            raise ModuleNotFoundError(msg)

        self.run_obj = run_obj
        self.deterministic = deterministic
        self.wallclock_limit = wallclock_limit
        self.kwargs = kwargs

        if facade in ["SMAC4BO", SMAC4BO, "SMAC4AC", SMAC4AC, "SMAC4HPO", SMAC4HPO, "BOHB4HPO", BOHB4HPO]:
            if type(facade) == str:

                self.facade = eval(facade)
            else:
                self.facade = facade
        else:
            msg = "SMAC.facade {} not known. Please use one of ['SMAC4BO', 'SMAC4AC', 'SMAC4HPO']."
            logger.error(msg.format(str(facade)))
            raise ValueError(msg.format(str(facade)))

        self.rng = rng
        if not intensifier_kwargs:
            self.intensifier_kwargs = {}
        else:
            self.intensifier_kwargs = intensifier_kwargs

        self.cspace = ConfigurationSpace()  # hyperparameter space for SMAC
        self.switch_optiones = {}
        self.hyperparameters = []

        self.maximize_metric = False
        self.constant_dictionary = {}

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function: Callable):
        """
        Initializes the SMAC Optimizer.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create hyperparameter space.

            maximize_metric:
                Boolean to distinguish between score and error.

            objective_function:
                The cost or objective function.

        """
        self.cspace = ConfigurationSpace()  # build space
        self._build_smac_space(pipeline_elements)
        if self.constant_dictionary:
            msg = "PHOTONAI has detected some one-valued params in your hyperparameters. Pleas use the kwargs for " \
                  "constant values. This run ignores following settings: " + str(self.constant_dictionary.keys())
            logger.warning(msg)
            warnings.warn(msg)
        self.maximize_metric = maximize_metric

        scenario_dict = self.kwargs
        scenario_dict.update({"run_obj":self.run_obj,
                              "deterministic":self.deterministic,
                              "wallclock_limit":self.wallclock_limit,
                              "cs":self.cspace,
                              "limit_resources": False})

        scenario = Scenario(scenario_dict)

        def smac_objective_function(current_config):
            current_config = {k: current_config[k] for k in current_config if (current_config[k] and 'algos' not in k)}
            return objective_function(current_config)

        self.smac = self.facade(scenario=scenario,
                                intensifier_kwargs=self.intensifier_kwargs,
                                rng=self.rng,
                                tae_runner=smac_objective_function)

    def optimize(self):
        """Start optimization process."""
        self.smac.optimize()

    def _build_smac_space(self, pipeline_elements: list):
        """
        Build the entire SMAC hyperparameter space.

        Parameters:
            pipeline_elements:
                List of all PipelineElements to create the hyperparameter space.

        """
        for pipe_element in pipeline_elements:
            # build conditions for switch elements
            if pipe_element.__class__.__name__ == 'Switch':
                algorithm_options = {}
                for algo in pipe_element.elements:
                    algo_params = []  # hyper params corresponding to "algo"
                    for name, value in algo.hyperparameters.items():
                        smac_param = self._convert_photonai_to_smac_param(value, (
                                pipe_element.name + "__" + name))  # or element.name__algo.name__name
                        algo_params.append(smac_param)
                    algorithm_options[(pipe_element.name + "__" + algo.name)] = algo_params

                algos = CategoricalHyperparameter(pipe_element.name + "__algos", choices=algorithm_options.keys())

                self.switch_optiones[pipe_element.name + "__algos"] = algorithm_options.keys()

                self.cspace.add_hyperparameter(algos)
                for algo, params in algorithm_options.items():
                    for param in params:
                        cond = InCondition(child=param, parent=algos, values=[algo])
                        self.cspace.add_hyperparameter(param)
                        self.cspace.add_condition(cond)
                continue

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
                    smac_param = self._convert_photonai_to_smac_param(value, name)
                    if smac_param is not None:
                        self.cspace.add_hyperparameter(smac_param)

    @staticmethod
    def _convert_photonai_to_smac_param(hyperparam: PhotonHyperparam, name: str):
        """
        Helper function: Convert PHOTONAI to SMAC hyperparameter.

        Parameters:
            hyperparam:
                One of photonai.optimization.hyperparameters.

            name:
                Name of hyperparameter.

        """
        if isinstance(hyperparam, PhotonCategorical) or isinstance(hyperparam, BooleanSwitch):
            return CategoricalHyperparameter(name, hyperparam.values)
        elif isinstance(hyperparam, list):
            return CategoricalHyperparameter(name, hyperparam)
        elif isinstance(hyperparam, FloatRange):
            if hyperparam.range_type in ['linspace', 'logspace']:
                return UniformFloatHyperparameter(name, hyperparam.start, hyperparam.stop,
                                                  log=(hyperparam.range_type == 'logspace'))
            msg = str(hyperparam.range_type) + "in your FloatRange is not implemented in SMAC."
            logger.error(msg)
            raise NotImplementedError(msg)
        elif isinstance(hyperparam, IntegerRange):
            return UniformIntegerHyperparameter(name, hyperparam.start, hyperparam.stop)

        msg = "Cannot convert hyperparameter " + str(hyperparam) + ". Supported types: Categorical, IntegerRange," \
                                                                   "FloatRange, list."
        logger.error(msg)
        raise ValueError(msg)
