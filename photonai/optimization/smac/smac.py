from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange
from photonai.optimization.base_optimizer import PhotonMasterOptimizer
from photonai.photonlogger import logger

try:
    from smac.configspace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
        ConfigurationSpace, Configuration, InCondition, Constant
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.facade.smac_ac_facade import SMAC4AC
    __found__ = True
except ModuleNotFoundError:
    __found__ = False


class SMACOptimizer(PhotonMasterOptimizer):

    def __init__(self, facade='SMAC4HPO', scenario_dict=None, intensifier_kwargs=None, rng=42, smac_helper=None):
        """
        SMAC Wrapper for PHOTON.
        SMAC usage and implementation details:
        https://automl.github.io/SMAC3/master/quickstart.html

        Parameters
        ----------
        * `facade` [str or smac.facade.class, default: 'SMAC4HPO']:
             Choice of SMAC backend strategy, [SMAC4BO, SMAC4HPO, SMAC4AC].
        * `scenario_dict` [dict, default: None (warning scenario with wallclock_limit = 60*40)]
            Informations for scenario settings like run_limit or wallclock_limit.
            Different to main SMAC cs (configspace) is not required or used cause PHOTON translate own param_space
            to SMAC.configspace.
        * `rng`: [int, default: 42]
            random seed of SMAC.facade
        * `smac_helper` [dict]
            For testing this object give public access to SMAC.facade object.
            Currently help object for test cases.
        """

        super(SMACOptimizer, self).__init__()

        if not __found__:
            msg = "Module smac not found or not installed as expected. " \
                  "Please install the smac_requirements.txt PHOTON provides."
            logger.error(msg)
            raise ModuleNotFoundError(msg)

        if not scenario_dict:
            self.scenario_dict = {"run_obj": "quality",
                                  "deterministic": "true",
                                  "wallclock_limit": 60 * 40}
            msg = "No scenario_dict for smac was given. Falling back to default values: {}.".format(self.scenario_dict)
            logger.warning(msg)
        else:
            self.scenario_dict = scenario_dict

        if facade in ["SMAC4BO", SMAC4BO, "SMAC4AC", SMAC4AC, "SMAC4HPO", SMAC4HPO]:
            if isinstance(facade, str):
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

        self.cspace = ConfigurationSpace()  # Hyperparameter space for SMAC
        self.switch_optiones = {}
        self.hyperparameters = []

        if smac_helper:
            self.smac_helper = smac_helper
            self.debug = True
        else:
            self.debug = False

        self.maximize_metric = False

    @staticmethod
    def _convert_photon_to_smac_param(hyperparam: object, name: str):
        """
        Helper function: Convert PHOTON hyperparameter to SMAC hyperparameter.

        Parameters
        ----------
        * `hyperparam` [object]:
             One of photonai.optimization.hyperparameters.
        * `name` [str]
            Name of hyperparameter.
        """
        if not hyperparam:
            return None
        if isinstance(hyperparam, PhotonCategorical):
            return CategoricalHyperparameter(name, hyperparam.values)
        elif isinstance(hyperparam, list):
            return CategoricalHyperparameter(name, hyperparam)
        elif isinstance(hyperparam, FloatRange):
            if hyperparam.range_type == 'linspace':
                return UniformFloatHyperparameter(name, hyperparam.start, hyperparam.stop)
            elif hyperparam.range_type == 'logspace':
                raise NotImplementedError("Logspace in your float hyperparameter is not implemented in SMAC.")
            else:
                return UniformFloatHyperparameter(name, hyperparam.start, hyperparam.stop)
        elif isinstance(hyperparam, IntegerRange):
            return UniformIntegerHyperparameter(name, hyperparam.start, hyperparam.stop)

    def build_smac_space(self, pipeline_elements):
        """
        Build entire SMAC hyperparameter space.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        """
        for pipe_element in pipeline_elements:
            # build conditions for switch elements
            if pipe_element.__class__.__name__ == 'Switch':
                algorithm_options = {}
                for algo in pipe_element.elements:
                    algo_params = []  # hyper params corresponding to "algo"
                    for name, value in algo.hyperparameters.items():
                        smac_param = self._convert_photon_to_smac_param(value, (
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
                    smac_param = self._convert_photon_to_smac_param(value, name)
                    if smac_param is not None:
                        self.cspace.add_hyperparameter(smac_param)

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objective_function):
        """
        Initializes SMAC Optimizer.

        Parameters
        ----------
        * `pipeline_elements` [list]:
            List of all pipeline_elements to create hyperparameter_space.
        * `maximize_metric` [bool]:
            Boolean for distinguish between score and error.
        * `objective_function` [callable]:
            The cost or objective function.
        """
        self.space = ConfigurationSpace()  # build space
        self.build_smac_space(pipeline_elements)
        self.maximize_metric = maximize_metric
        self.scenario_dict["cs"] = self.cspace
        self.scenario_dict["limit_resources"] = False

        self.scenario = Scenario(self.scenario_dict)

        def smac_objective_function(current_config):
            current_config = {k: current_config[k] for k in current_config if (current_config[k] and 'algos' not in k)}
            return objective_function(current_config)

        self.smac = self.facade(scenario = self.scenario,
                                intensifier_kwargs = self.intensifier_kwargs,
                                rng = self.rng,
                                tae_runner = smac_objective_function)

        if self.debug:
            self.smac_helper['data'] = self.smac

    def optimize(self):
        """
        Start optimization process.
        """
        self.smac.optimize()
