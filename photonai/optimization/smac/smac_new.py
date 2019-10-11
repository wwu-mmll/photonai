import numpy as np

from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange
from photonai.optimization.base_optimizer import PhotonBaseOptimizer
from photonai.optimization.smac.execute_ta_run import MyExecuteTARun

try:
    from smac.configspace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
        ConfigurationSpace, Configuration, InCondition, Constant
    from smac.scenario.scenario import Scenario
    from smac.tae.execute_ta_run import StatusType
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.intensification.intensification import Intensifier
    __found__ = True
except ModuleNotFoundError:
    __found__ = False


class SMACOptimizer(PhotonBaseOptimizer):

    def __init__(self,
                 run_obj="quality",
                 wallclock_limit: float = float("inf"),
                 min_r=1,
                 max_r=1,
                 run_limit=40):
        if __found__:
            self.wallclock_limit = wallclock_limit
            self.minR = min_r
            self.maxR = max_r
            if self.minR != 1 or self.maxR != 1:
                raise NotImplementedError("PHOTONs seed management is not implemented yet. "
                                          "At this juncture we can not run a config multiple times correctly.")

            self.cspace = ConfigurationSpace()  # Hyperparameter space for smac
            self.runtime = 0
            self.run_start = 0
            self.run_obj = run_obj
            self.run_limit = run_limit

            self.challengers = []
            self.old_challengers = None
            self.ask_list = []

            self.budget_exhausted = False
            self.constant_dictionary = {}

        else:
            raise ModuleNotFoundError("Module smac not found or not installed as expected. "
                                      "Please install the smac_requirements.txt PHOTON provides.")

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
            return UniformIntegerHyperparameter(name, hyperparam.start, hyperparam.stop-1)

    def build_smac_space(self, pipeline_elements):
        """
        Build entire smac hyperparam space.
        """
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
                    smac_param = self._convert_photon_to_smac_space(value, name)
                    if smac_param is not None:
                        self.cspace.add_hyperparameter(smac_param)

    def prepare(self, pipeline_elements: list, maximize_metric: bool):
        # build space
        self.space = ConfigurationSpace()

        self.build_smac_space(pipeline_elements)

        self.scenario = Scenario({"run_obj": self.run_obj,
                                  "cs": self.cspace,
                                  "deterministic": "true",
                                  "wallclock_limit": self.wallclock_limit})

        self.smac = SMAC4BO(scenario=self.scenario,
                            rng=np.random.RandomState(42),
                            tae_runner=MyExecuteTARun)
        self.optimizer = self.smac.solver
        self.optimizer.runhistory.overwrite_existing_runs = True
        self.ask = self.ask_generator()

    def tell(self, config, performance):

        if not config:
            return

        config = Configuration(self.cspace, values=config)

        self.optimizer.stats.ta_runs += 1

        # first incubment setting
        if self.runtime == 0:
            self.optimizer.incumbent = Configuration(self.cspace, values=self.challengers[0])
            self.optimizer.start()
        else:
            if self.optimizer.stats.is_budget_exhausted():
                self.budget_exhausted = True

        self.optimizer.runhistory.add(config=config, cost=performance[1], time=0, status=StatusType.SUCCESS,
                                      instance_id="", seed=5)

        self.runtime += 1

        if self.runtime % 20 == 0:
            print("test")

    def ask_generator(self):
        def init():
            ta_run = MyExecuteTARun(run_limit=self.run_limit, runhistory=self.optimizer.runhistory)
            self.optimizer.intensifier = Intensifier(tae_runner=ta_run,
                                                     stats=self.optimizer.stats,
                                                     traj_logger=None,
                                                     rng=np.random.RandomState(42),
                                                     instances=list(self.optimizer.runhistory.ids_config.keys()),
                                                     minR=1, maxR=1,
                                                     run_limit=self.run_limit)

            X, Y = self.optimizer.rh2EPM.transform(self.optimizer.runhistory)

            self.optimizer.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.optimizer.choose_next(X, Y)

            # first run challenger vs. any other run
            if isinstance(challengers, list):
                self.challengers = challengers[:self.run_limit]
            else:
                self.challengers = challengers.challengers[:self.run_limit]

            self.ask_list = [x.get_dictionary() for x in self.challengers]
            if self.old_challengers:
                self.incumbent, inc_perf = self.optimizer.intensifier.\
                    intensify(challengers=self.old_challengers,
                              incumbent=self.optimizer.incumbent,
                              run_history=self.optimizer.runhistory,
                              aggregate_func=self.optimizer.aggregate_func,
                              log_traj=False,
                              time_bound=max(self.optimizer.intensifier._min_time, 60))
                self.optimizer.incumbent = self.incumbent
            self.old_challengers = self.challengers

            return 0

        i = init()
        while True:
            if self.budget_exhausted:
                val = yield {}
                return
            else:
                val = (yield self.ask_list[i])
            if len(self.ask_list) - 1 == i:
                i = init()
            else:
                i += 1
