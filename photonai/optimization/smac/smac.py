from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange
from photonai.optimization.base_optimizer import PhotonMasterOptimizer
from photonai.photonlogger import logger

import logging

#logging.basicConfig(level=logging.DEBUG)

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

    def __init__(self, scenario_dict, intensifier_kwargs = None, rng = 42, smac_helper = None):
        """
        SMAC Wrapper for PHOTON.
        SMAC usage and implementation details:
        https://automl.github.io/SMAC3/master/quickstart.html

        :param scenario_dict: dict for scenario settings
        :param rng: INT for random seed
        :param smac_helper: currently help object for test cases.
        """

        if not __found__:
            msg = "Module smac not found or not installed as expected. " \
                  "Please install the smac_requirements.txt PHOTON provides."
            logger.error(msg)
            raise ModuleNotFoundError("Module smac not found or not installed as expected. "
                                      "Please install the smac_requirements.txt PHOTON provides.")

        if not scenario_dict:
            msg = "Please enter scenario_dict for smac intern informations."
            logger.error(msg)
            raise ValueError(msg)

        self.rng = rng
        if not intensifier_kwargs:
            self.intensifier_kwargs = {}
        else:
            self.intensifier_kwargs = intensifier_kwargs

        self.cspace = ConfigurationSpace()  # Hyperparameter space for smac

        self.scenario_dict = scenario_dict

        self.switch_optiones = {}
        self.hyperparameters = []

        if smac_helper:
            self.smac_helper = smac_helper
            self.debug = True
        else:
            self.debug = False

        self.maximize_metric = False


    @staticmethod
    def _convert_photon_to_smac_space(hyperparam: object, name: str):
        """
        Helper function: Convert PHOTON hyperparams to smac params.
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
        Build entire smac hyperparam space.
        """
        for pipe_element in pipeline_elements:
            # build conditions for switch elements
            if pipe_element.__class__.__name__ == 'Switch':
                algorithm_options = {}
                for algo in pipe_element.elements:
                    algo_params = []  # hyper params corresponding to "algo"
                    for name, value in algo.hyperparameters.items():
                        smac_param = self._convert_photon_to_smac_space(value, (
                                pipe_element.name + "__" + name))  # or element.name__algo.name__name
                        algo_params.append(smac_param)
                    algorithm_options[(pipe_element.name + "__" + algo.name)] = algo_params

                algos = CategoricalHyperparameter(name=pipe_element.name + "__algos", choices=algorithm_options.keys())

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
                    smac_param = self._convert_photon_to_smac_space(value, name)
                    if smac_param is not None:
                        self.cspace.add_hyperparameter(smac_param)

    def prepare(self, pipeline_elements: list, maximize_metric: bool, objectiv_function):
        """
        Initializes SMAC Optimizer.
        :param pipeline_elements: PHOTON Elments to cast
        :param maximize_metric: error or score distinction
        :param objectiv_function: function cfg -> cost
        :return:
        """
        self.space = ConfigurationSpace()  # build space
        self.build_smac_space(pipeline_elements)
        self.maximize_metric = maximize_metric
        self.scenario_dict["cs"] = self.cspace
        self.scenario_dict["limit_resources"] = False

        self.scenario = Scenario(self.scenario_dict)

        self.smac = SMAC4BO(scenario = self.scenario,
                            intensifier_kwargs = self.intensifier_kwargs,
                             rng = self.rng,
                             tae_runner = objectiv_function)

        #self.smac.logger.basicConfig(level=logging.DEBUG)

        if  self.debug:
            self.smac_helper['data'] = self.smac


    def optimize(self):
        """
        Start optimization process
        """
        self.smac.optimize()
