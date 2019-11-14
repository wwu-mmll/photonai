import time

import numpy as np

from photonai.base import Switch
from photonai.optimization import Categorical as PhotonCategorical
from photonai.optimization import FloatRange, IntegerRange, BooleanSwitch
from photonai.optimization.base_optimizer import PhotonBaseOptimizer
from photonai.photonlogger.logger import logger

try:
    from smac.configspace import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, \
        ConfigurationSpace, Configuration, InCondition, Constant
    from smac.scenario.scenario import Scenario
    from smac.tae.execute_ta_run import StatusType
    # from smac.facade.smac_facade import SMAC, SMBO
    from smac.facade.smac_bo_facade import SMAC4BO, SMAC4AC
except ModuleNotFoundError:
    raise ModuleNotFoundError("Module smac not found or not installed as expected. "
                              "Please install the smac_requirements.txt PHOTON provides.")


class SMACOptimizer(PhotonBaseOptimizer):

    def __init__(self, cutoff_time: float = float('inf'),
                 run_obj: str = "quality",
                 runcount_limit: float = float("inf"),
                 tuner_timeout: float = float("inf"),
                 wallclock_limit: float = float("inf")):
        self.optimizer = None
        self.scenario = None
        self.hyperparameter_list = []
        self.metric_to_optimize = ''
        if (run_obj == "runtime" and cutoff_time == float('inf')) or run_obj == "quality":
            cutoff_time = None
        self.ask = self.ask_generator()
        self.runcount_limit = runcount_limit  # should there be something more? Refer - https://automl.github.io/SMAC3/stable/options.html
        self.tuner_timeout = tuner_timeout
        self.wallclock_limit = wallclock_limit
        self.run_obj = run_obj
        self.cutoff_time = cutoff_time
        # self.maximize_metric = True
        self.memory = {}  # stores config : cost
        self.test = 0

    def prepare(self, pipeline_elements: list, maximize_metric: bool):

        self.hyperparameter_list = []
        # build space
        self.space = ConfigurationSpace()

        for element in pipeline_elements:
            # check if Switch object

            if isinstance(element, Switch):
                algorithm_options = {}  # mapping algorithm name with their child hyper params

                for algo in element.elements:
                    algo_params = []  # hyper params corresponding to "algo"
                    for name, value in algo.hyperparameters.items():
                        smac_param = self._convert_PHOTON_to_smac_space(value, (
                                element.name + "__" + name))  # or element.name__algo.name__name ???
                        algo_params.append(smac_param)
                    algorithm_options[(element.name + "__" + algo.name)] = algo_params

                algos = CategoricalHyperparameter(name=element.name + "__algos", choices=algorithm_options.keys())
                self.space.add_hyperparameter(algos)
                for algo, params in algorithm_options.items():
                    for param in params:
                        cond = InCondition(child=param, parent=algos, values=[algo])
                        self.space.add_hyperparameter(param)
                        self.space.add_condition(cond)


            else:
                for name, value in element.hyperparameters.items():
                    smac_param = self._convert_PHOTON_to_smac_space(value, name)
                    if smac_param is not None:
                        self.space.add_hyperparameter(smac_param)

        self.scenario = Scenario({"run_obj": self.run_obj,
                                  "cutoff_time": self.cutoff_time,
                                  "runcount_limit": self.runcount_limit,
                                  "tuner-timeout": self.tuner_timeout,
                                  "wallclock_limit": self.wallclock_limit,
                                  "cs": self.space,
                                  "deterministic": "true"
                                  })
        self.smac = SMAC4BO(scenario=self.scenario, rng=np.random.RandomState(42))
        self.optimizer = self.smac.solver
        self.optimizer.stats.start_timing()
        self.optimizer.incumbent = self.get_default_incumbent()

        self.flag = False  # False: compute performance of challenger, True: compute performance of incumbent
        self.ask = self.ask_generator()

    def get_default_incumbent(self):
        instantiated_hyperparameters = {}
        for hp in self.space.get_hyperparameters():
            conditions = self.space._get_parent_conditions_of(hp.name)
            active = True
            for condition in conditions:
                parent_names = [c.parent.name for c in
                                condition.get_descendant_literal_conditions()]

                parents = {
                    parent_name: instantiated_hyperparameters[parent_name]
                    for parent_name in parent_names
                }

                if not condition.evaluate(parents):
                    # TODO find out why a configuration is illegal!
                    active = False

            if not active:
                instantiated_hyperparameters[hp.name] = None
            elif isinstance(hp, Constant):
                instantiated_hyperparameters[hp.name] = hp.value
            else:
                instantiated_hyperparameters[hp.name] = hp.default_value

        config = Configuration(self.space, instantiated_hyperparameters)
        return (config)

    def _convert_PHOTON_to_smac_space(self, hyperparam: object, name: str):
        if not hyperparam:
            return None
        self.hyperparameter_list.append(name)
        if isinstance(hyperparam, PhotonCategorical):
            return CategoricalHyperparameter(choices=hyperparam.values, name=name)
        elif isinstance(hyperparam, list):
            return CategoricalHyperparameter(choices=hyperparam, name=name)
        elif isinstance(hyperparam, FloatRange):
            return UniformFloatHyperparameter(lower=hyperparam.start, upper=hyperparam.stop, name=name)
        elif isinstance(hyperparam, IntegerRange):
            return UniformIntegerHyperparameter(lower=hyperparam.start, upper=hyperparam.stop, name=name)
        elif isinstance(hyperparam, BooleanSwitch):
            return CategoricalHyperparameter(choices=["true", "false"], name=name)

    def ask_generator(self):
        while True:
            self.flag = False
            start_time = time.time()

            X, Y = self.optimizer.rh2EPM.transform(self.optimizer.runhistory)

            self.optimizer.logger.debug("Search for next configuration.")
            # get all configurations sorted according to acquision function
            challengers = self.optimizer.choose_next(X, Y)
            self.test += 1
            print("TEST # of trains", self.test)
            time_spent = time.time() - start_time
            time_left = self.optimizer._get_timebound_for_intensification(time_spent)

            self.to_run = self.intensify(
                challengers=challengers,
                incumbent=self.optimizer.incumbent,
                run_history=self.optimizer.runhistory,
                aggregate_func=self.optimizer.aggregate_func,
                time_bound=max(self.optimizer.intensifier._min_time, time_left))

            if self.flag:
                if self.optimizer.stats.is_budget_exhausted():
                    # yield self.optimizer.incumbent.get_dictionary()
                    return None
                else:
                    yield self.check(self.to_run.get_dictionary())

            else:
                print("Size of challenger list: ", len(self.to_run))
                for challenger in self.to_run[:min(len(self.to_run), 25)]:
                    if self.optimizer.stats.is_budget_exhausted():
                        # yield self.optimizer.incumbent.get_dictionary()
                        return None
                    else:
                        yield self.check(challenger.get_dictionary())

            logger.debug(
                "Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)"
                % (self.optimizer.stats.get_remaing_time_budget(),
                   self.optimizer.stats.get_remaining_ta_budget(),
                   self.optimizer.stats.get_remaining_ta_runs()))

            self.optimizer.stats.print_stats(debug_out=True)

    def check(self, config):
        config_dict = {}
        self.flag0 = False
        for k, val in config.items():
            if not 'algos' in k:
                config_dict[k] = val
            else:
                self.temp = (k, val)
                self.flag0 = True
        return config_dict

    def intensify(self, challengers, incumbent, run_history, aggregate_func, time_bound):
        instance_specifics = {}
        _num_run = 0
        _chall_indx = 0
        _ta_time = 0
        _min_time = 10 ** -5
        start_time = time.time()
        to_run = None

        if time_bound < _min_time:
            raise ValueError("time_bound must be >= %f" % (_min_time))

        to_run_configs = []
        for challenger in challengers:
            if challenger == incumbent:
                continue

            if self.hashable_dict(incumbent.get_dictionary()) not in self.memory:
                self.flag = True
                return incumbent

            to_run = self.race_challenger(challenger=challenger)
            if to_run is not None:
                to_run_configs.append(to_run)

        return to_run_configs

    def race_challenger(self, challenger):
        if self.hashable_dict(challenger.get_dictionary()) not in self.memory:
            return challenger
        else:
            return None

    def compare_configs(self, incumbent, challenger):
        inc_perf = self.memory[self.hashable_dict(incumbent.get_dictionary())]
        chal_perf = self.memory[self.hashable_dict(challenger.get_dictionary())]

        if inc_perf > chal_perf:
            return incumbent
        else:
            return challenger

    def hashable_dict(self, config: dict):
        return frozenset(config.items())

    def tell(self, config: dict, performance: float, runtime: float = 2.0):
        if self.flag0:
            config[self.temp[0]] = self.temp[1]

        performance = performance[1]
        self.optimizer.stats.ta_runs += 1
        self.optimizer.stats.ta_time_used += runtime

        hash_config = self.hashable_dict(config)
        self.memory[hash_config] = performance

        # convert dictionary to list in correct order
        config = Configuration(self.space, values=config)
        if self.run_obj == "runtime":
            performance = -runtime
            if runtime > self.cutoff_time:
                runtime = self.cutoff_time
                self.optimizer.runhistory.add(config=config, cost=performance, time=runtime, status=StatusType.TIMEOUT)
            else:
                self.optimizer.runhistory.add(config=config, cost=performance, time=runtime, status=StatusType.SUCCESS)
        else:
            self.optimizer.runhistory.add(config=config, cost=performance, time=runtime, status=StatusType.SUCCESS)

        if self.flag:
            pass
        else:
            self.optimizer.incumbent = self.compare_configs(self.optimizer.incumbent, config)
