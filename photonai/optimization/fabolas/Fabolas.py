import numpy as np
import george

import os
import json
from copy import copy
from time import time

from photonai.optimization.fabolas.GPMCMC import FabolasGPMCMC
from photonai.optimization.fabolas.Priors import EnvPrior
from photonai.optimization.fabolas.Maximizer import InformationGainPerUnitCost, Direct, MarginalizationGPMCMC
from photonai.photonlogger import Logger
from photonai.optimization import FloatRange, IntegerRange, Categorical


def _quadratic_bf(x):
    '''
    Quadratic base-function for the model_objective

    Can't be a lambda because of parallelization
    :param x: x
    :return: (1-x)**2
    '''
    return (1 - x) ** 2


def _linear_bf(x):
    '''
    Linear base-function the mode_cost

    Can't be a lambda because of parallelization
    :param x: x
    :return: x
    '''
    return x


class Fabolas:
    def __init__(
            self,
            n_min_train_data,
            n_train_data,
            pipeline_elements,
            n_init=40,
            num_iterations=100,
            subsets=[256, 128, 64],
            burnin=100,
            chain_length=100,
            n_hypers=12,
            model_pool_size=-1, # if -1 then min(n_hypers, cpu_count)
            acquisition_pool_size=-1,
            rng=None,
            verbose_maximizer=False,
            verbose_gp=False,
            consider_opt_time=False,
            log=None,
            maximizer_func_evals=200,
            **kwargs
    ):
        '''
        Initializes the Fabolas-Object.

        Creates and initializes the models and the maximizer. Reads the hyperparameters and their boundaries.
        :param n_min_train_data: Minimal amount of data to use for training
        :type n_min_train_data: int
        :param n_train_data: Maximum amount of data to use for training (normally the size of the full set)
        :type n_train_data: int
        :param pipeline_elements: Elements that need to be optimized
        :type pipeline_elements: iterable of PipelineElement
        :param n_init: Number of iterations in the initizialization-phase
        :type n_init: int
        :param num_iterations: Total number of iterations
        :type num_iterations: int
        :param subsets: Subset-fragmentations to use in the init-phase
        :type subsets: iterable of int
        :param burnin: burning-elements of the MCMC-models
        :type burnin: int
        :param chain_length: chain-length of the MCMC-models
        :type chain_length: int
        :param n_hypers: number of hyper-parameters used by the MCMC-models
        :type n_hypers: int
        :param model_pool_size: Number of threads to use for all models
        :type model_pool_size: int
        :param acquisition_pool_size: Number of threads to use for the maximizer
        :type acquisition_pool_size: int
        :param rng: Random number generator. Creates a new one if null
        :type rng: np.random.RandomState
        :param verbose_maximizer: Print maximizer-output to screen
        :type verbose_maximizer: bool
        :param verbose_gp: Print hyperparameter-configurations of the MCMC-Models while optimization
        :type verbose_gp: bool
        :param consider_opt_time:
            if true: use the calculation-time + evaluation-time as cost (as proposed in the FABOLAS-Paper)
            else: use the evaluation-time as cost (as in the original source code)
        :type consider_opt_time: bool
        :param log:
            if null: don't log results in each iteration
            else: following dict:
                id: process id for simultaneous calculations
                name: name of your pipe/calculation
                path: path to a directory where to store the folder containing the logfiles
                incumbents:
                    if true: calculate optimal config (incumbents) in each iteration and log them (needs additional computation time)
                    else: don't log incumbents
        :type log: dict
        :param maximizer_func_evals: Number of function-evaluations for the maximizer
        :type maximizer_func_evals: int
        :param kwargs: unused (catchall for named params)
        :type kwargs: dict
        '''
        assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

        if log is not None:
            if not isinstance(log, dict):
                raise ValueError("log must be a dict with keys id, path, name")
            log['id'] = int(log['id']) if 'id' in log else 0
            log['name'] = str(log['name']) if 'name' in log else 'fabolas'
            log['incumbents'] = bool(log['incumbents']) if 'incumbents' in log else False
            if 'path' not in log:
                raise ValueError("log must contain the key path")
            log['bn'] = '{name}_{id}'.format(name=log['name'], id=log['id'])
            log['path'] = os.path.realpath(os.path.join(str(log['path']), log['bn']))
            if not os.path.exists(log['path']):
                os.makedirs(log['path'])
            Logger().info("Fabolas: writing logs to "+log['path'])

        self._log = log
        self._verbose_maximizer = verbose_maximizer
        self._lower = []
        self._upper = []
        self._number_param_keys = []
        self._param_types = []
        self._param_dict = {}
        self.tracking = None
        self.s = None

        # Todo: Edit to list, FloatRange, Categorial and IntegerRange
        for pelem in pipeline_elements:
            for key, val in pelem.hyperparameters.items():
                # key = elements[0].name + '__' + key
                if val:
                    self._number_param_keys.append(key)

                    if val is bool:
                        val = [bool, 0, 1]

                    # if we have a list that uses original fabolas hyperparam specification [type, lower, upper]
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], type):
                        lo = 0
                        up = 0
                        if val[0] is bool:
                            lo = 0
                            up = 1

                        if val[0] in [float, int]:
                            if len(val) < 3 or val[1] > val[2]:
                                raise ValueError(
                                    "Error for param '" + key + "'."
                                    "First value must be lower bound, second value must be upper bound."
                                )
                            lo = val[1]
                            up = val[2]

                        if val[0] is str:
                            if len(val) < 3:
                                raise ValueError(
                                    'Please provide at least two strings or use "param=\'mystring\''
                                )
                            lo = 0
                            up = np.log(len(val)-1)

                        self._lower.append(lo)
                        self._upper.append(up)
                        self._param_types.append(val[0])

                    # if we have a list that contains distinct values like [0, 1, 2, 3, 4]
                    if isinstance(val, list) and len(val) > 0 and not isinstance(val[0], type):
                        if isinstance(val[0], str):
                            self._lower.append(0)
                            self._upper.append(np.log(len(val) - 1))
                            self._param_types.append(str)
                        elif isinstance(val[0], bool):
                            self._lower.append(0)
                            self._upper.append(1)
                            self._param_types.append(bool)
                        elif type(val[0]) in [int, float]:
                            sorted_list = np.sort(val)
                            self._lower.append(sorted_list[0])
                            self._upper.append(sorted_list[-1])
                            self._param_types.append(type(val[0]))

                    # if we have a Photon Number Representation or a Photon Categorical representation
                    if isinstance(val, IntegerRange) or isinstance(val, FloatRange):
                        self._lower.append(val.start)
                        self._upper.append(val.stop)
                        # self._upper.append(np.log(val.stop))
                        if isinstance(val, IntegerRange):
                            self._param_types.append(int)
                        elif isinstance(val, FloatRange):
                            self._param_types.append(float)
                    if isinstance(val, Categorical):
                        self._lower.append(0)
                        self._upper.append(np.log(len(val.values)-1))
                        self._param_types.append(str)
                    self._param_dict.update({key: val})

        n_dims = len(self._lower)
        self._lower = np.array(self._lower)
        self._upper = np.array(self._upper)

        self._s_min = n_min_train_data
        self._s_max = n_train_data
        self._n_init = n_init
        self._num_iterations = num_iterations
        self._subsets = subsets
        self._rng = np.random.RandomState() if rng is None else rng
        self._consider_opt_time = consider_opt_time

        self._X = []
        self._Y = []
        self._cost = []
        self._it = 0
        self._model_objective = None
        self._model_cost = None

        kernel = 1  # 1 = covariance amplitude

        # ARD Kernel for the configuration space
        degree = 1
        for d in range(n_dims):
            kernel *= george.kernels.Matern52Kernel(
                np.ones([1]) * 0.01,
                ndim=n_dims+1,
                # dim=d
            )

        env_kernel = george.kernels.LinearKernel(
            n_dims+1,
            ndim=n_dims+1,
            order=degree
        )
        # env_kernel[:] = np.ones([degree + 1]) * 0.1

        kernel *= env_kernel

        # Take 3 times more samples than we have hyperparameters
        if n_hypers < 2*len(kernel):
            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

        prior = EnvPrior(
            len(kernel) + 1,
            n_ls=n_dims,
            n_lr=(degree + 1),
            rng=self._rng
        )

        self._model_objective = FabolasGPMCMC(
            kernel,
            prior=prior,
            burnin_steps=burnin,
            chain_length=chain_length,
            n_hypers=n_hypers,
            normalize_output=False,
            basis_func=_quadratic_bf,
            lower=self._lower,
            upper=self._upper,
            rng=self._rng,
            pool_size=model_pool_size,
            verbose_gp=verbose_gp
        )

        cost_degree = 1
        cost_env_kernel = george.kernels.LinearKernel(
            n_dims+1,
            ndim=n_dims+1,
            order=cost_degree
        )
        cost_kernel = 1  # 1 = covariance amplitude

        # ARD Kernel for the configuration space
        for d in range(n_dims):
            cost_kernel *= george.kernels.Matern52Kernel(
                np.ones([1]) * 0.01,
                ndim=n_dims+1,
                # dim=d
            )

        # cost_env_kernel[:] = np.ones([cost_degree + 1]) * 0.1

        cost_kernel *= cost_env_kernel

        cost_prior = EnvPrior(
            len(cost_kernel) + 1,
            n_ls=n_dims,
            n_lr=(cost_degree + 1),
            rng=rng
        )

        self._model_cost = FabolasGPMCMC(
            cost_kernel,
            prior=cost_prior,
            burnin_steps=burnin,
            chain_length=chain_length,
            n_hypers=n_hypers,
            basis_func=_linear_bf,
            normalize_output=False,
            lower=self._lower,
            upper=self._upper,
            rng=self._rng,
            pool_size=model_pool_size,
            verbose_gp=verbose_gp
        )

        # Extend input space by task variable
        extend_lower = np.append(self._lower, 0)
        extend_upper = np.append(self._upper, 1)
        is_env = np.zeros(extend_lower.shape[0])
        is_env[-1] = 1

        # Define acquisition function and maximizer
        ig = InformationGainPerUnitCost(
            self._model_objective,
            self._model_cost,
            extend_lower,
            extend_upper,
            is_env_variable=is_env,
            n_representer=50,
            verbose=verbose_gp
        )
        self._acquisition_func = MarginalizationGPMCMC(ig, pool_size=acquisition_pool_size)

        direct_logfile = os.devnull if self._log is None\
            else os.path.join(self._log['path'], "maximizer_results.txt")

        self._maximizer = Direct(
            self._acquisition_func,
            extend_lower,
            extend_upper,
            verbose=self._verbose_maximizer,
            logfilename=direct_logfile,
            n_func_evals=maximizer_func_evals
        )

    def calc_config(self):
        '''
            Calculates the configurations and the subset-fragmentation to evaluate.
            Implemented as a generator.

            The returned tracking vars are for internal use and need to be passed to process_result.
        :return: next configuration to test, subset-frag to use, tracking-vars
        :rtype: dict, int, dict
        '''
        Logger().info('**Fabolas: Starting initialization')
        for self._it in range(0, self._n_init):
            Logger().debug('Fabolas: step ' + str(self._it) + ' (init)')
            start = time()
            result = self._init_models()
            tracking = {'overhead_time': time()-start}
            Logger().debug('Fabolas: needed {t!s}s'.format(t=tracking['overhead_time']))
            yield self._create_param_dict(result, tracking)

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._cost = np.array(self._cost)

        Logger().info('**Fabolas: Starting optimization')
        for self._it in range(self._n_init, self._num_iterations):
            Logger().debug('Fabolas: step ' + str(self._it) + ' (opt)')
            start = time()
            result = self._optimize_config()
            tracking = {'overhead_time': time()-start}
            Logger().debug('Fabolas: needed {t!s}s'.format(t=tracking['overhead_time']))
            yield self._create_param_dict(result, tracking)

        Logger().info('Fabolas: Final config')
        start = time()
        self._model_objective.train(self._X, self._Y, do_optimize=True)
        result = self.get_incumbent()
        tracking = {'overhead_time': time()-start}
        Logger().debug('Fabolas: needed {t!s}s'.format(t=tracking['overhead_time']))
        yield self._create_param_dict(result, tracking)

    def process_result(self, config, score, cost):
        '''
        Process results from the calculation

        :param config: used configuration
        :type config: dict
        :param subset_frac: used subset-fragmentation
        :type subset_frac: int
        :param score: evaluation loss
        :type score: float
        :param cost: evaluation cost
        :type cost: float
        :param tracking_vars: tracking vars returned by calc_config
        :type tracking_vars: dict
        '''

        tracking_vars = self.tracking
        subset_frac = self.s
        # We're done
        if self._it >= self._num_iterations:
            return

        score = 1-score

        if self._consider_opt_time:
            cost += tracking_vars['overhead_time']

        config_dict = config # preserve for logging

        # init-loop
        if self._it < self._n_init:
            config = self._get_params_from_dict(config)
            self._X.append(np.append(config, self._transform(self._s_max/subset_frac)))
            self._Y.append(np.log(score))  # Model the target function on a logarithmic scale
            self._cost.append(np.log(cost))  # Model the cost on a logarithmic scale

        # opt-loop
        elif self._n_init <= self._it < self._num_iterations:
            config = np.array(self._get_params_from_dict(config))
            config = np.append(config, self._transform(self._s_max/subset_frac))
            self._X = np.concatenate((self._X, config[None, :]), axis=0)
            self._Y = np.concatenate((self._Y, np.log(np.array([score]))), axis=0)  # Model the target function on a logarithmic scale
            self._cost = np.concatenate((self._cost, np.log(np.array([cost]))), axis=0)  # Model the cost function on a logarithmic scale

        self._generate_log(config_dict, subset_frac, score, cost, tracking_vars)

    def get_incumbent(self):
        '''
            Calculates the best configuration that can be calculated from known data (incumbent)
        :return: Configuration, subset-fragmentation = 1
        :rtype: list of numbers, int
        '''
        # This final configuration should be the best one
        final_config, _ = self._projected_incumbent_estimation(
            self._model_objective, self._X[:, :-1], proj_value=1
        )
        return final_config[:-1].tolist(), 1  # subset is the whole data-set

    def _adjust_param_types(self, params):
        '''
        Rounds all hyperparameters that have to be integers and casts them to int

        :param params: all hyperparameters as list
        :type params: list

        :return: alls hyperparameters with corrected types
        :rtype: list
        '''
        paramlist = params.tolist()
        for i, type in enumerate(self._param_types):
            if type is bool:
                paramlist[i] = bool(np.round(paramlist[i]))
            elif type is int:
                paramlist[i] = int(np.round(paramlist[i]))
            elif type is str:
                choices = self._param_dict[self._number_param_keys[i]].values
                paramlist[i] = np.round(np.exp(paramlist[i]))
                paramlist[i] = max(paramlist[i], 1)
                paramlist[i] = min(paramlist[i], len(choices)-1)
                paramlist[i] = str(choices[int(paramlist[i])])
        return paramlist

    def _create_param_dict(self, params, tracking=None):
        '''
        Create the parameter-dictionary from the internally used parameter-list and splits the subset-fragmentation

        :param params: hyperparameters as list
        :type params: list
        :param tracking: tracking-vars for Fabolas (pass-thru)
        :type tracking: dict

        :return: hyperparameters as dict, subset-fragmentation, tracking
        :rtype: dict, int, dict
        '''
        params, s = params
        if tracking is not None:
            tracking.update({'config_log': dict(zip(self._number_param_keys, params))})

        # exp_params = np.exp(params)
        adjusted_params = self._adjust_param_types(params)
        pdict = copy(self._param_dict)
        pdict.update(
            dict(zip(
                self._number_param_keys,
                adjusted_params
            ))
        )

        self.tracking = tracking
        self.s = s
        # was return pdict, tracking, s
        return pdict

    def _get_special_params(self):
        return {'subset_frac': self.s}

    def _get_params_from_dict(self, pdict):
        '''
        Turn the hyperparameter-dictionary to the internally used hyperparameter-list

        :param pdict: hyperparameter-dictionary
        :type pdict: dict

        :return: hyperparameter-list
        :rtype: list
        '''
        params = []
        for i, key in enumerate(self._number_param_keys):
            if self._param_types[i] is bool:
                pdict[key] = int(pdict[key])
            if self._param_types[i] is str:
                pdict[key] = np.log(self._param_dict[key].index(pdict[key]))
            params.append(pdict[key])
        return params

    def _init_models(self):
        '''
        Calculate configuration and subset-fragmentation in the init-phase

        :return: configuration, subset-fragmentation
        :rtype: list, int
        '''
        s = self._subsets[self._it % len(self._subsets)]
        x = self._init_random_uniform(self._lower, self._upper, 1)[0]
        return x, s

    def _optimize_config(self):
        '''
        Train models and calculate the configuration and subset-fragmentation in the optimization-phase

        :return: configuration, subset-fragmentation
        :rtype: list, int
        '''
        # Train models
        Logger().debug("Fabolas: Train model_objective")
        self._model_objective.train(self._X, self._Y, do_optimize=True)
        Logger().debug("Fabolas: Train model_cost")
        self._model_cost.train(self._X, self._cost, do_optimize=True)

        # Maximize acquisition function
        Logger().debug("Fabolas: Update acquisition func")
        self._acquisition_func.update(self._model_objective, self._model_cost)
        Logger().debug("Fabolas: Generate new config by maximizing")
        new_x = self._maximizer.maximize()

        s = self._s_max/self._retransform(new_x[-1])
        Logger().debug("Fabolas: config generation done for this step")

        return new_x[:-1], int(s)

    def _projected_incumbent_estimation(self, model, X, proj_value=1):
        '''
        Calculate the incumbent by projection

        :param model: trained model
        :type model: Model object
        :param X: hyperparameters
        :type X: list
        :param proj_value: projection-value
        :type proj_value: float

        :return: incumbent, assumed error
        :rtype: list, float
        '''
        projection = np.ones([X.shape[0], 1]) * proj_value
        X_projected = np.concatenate((X, projection), axis=1)

        m, _ = model.predict(X_projected)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value

    def _generate_log(self, conf, subset, result, cost, tracking_vars):
        '''
        Generates the log and stores it into the logfile and calculate the incumbent if the constructor-parameter log['incumbent'] was true

        :param conf: used configuration
        :type conf: dict
        :param subset: used subset-fragmentation
        :type subset: int
        :param result: the result of the evaluation
        :type result: float
        :param cost: the cost of the evaluation
        :type cost: float
        :param tracking_vars: Fabolas' tracking vars
        :type tracking_vars: dict
        '''
        if self._log is None:
            return

        Logger().debug("Fabolas: generating log")
        l = {
            'config': conf,
            'subset_frac': subset,
            'config_result': result,
            'config_cost': cost,
            'iteration': self._it,
            'operation': 'init' if self._it < self._n_init else 'opt'
        }
        if self._it == self._num_iterations:
            l['operation'] = 'final'

        if self._log['incumbents'] and self._it < self._num_iterations:
            start = time()
            if self._it < self._n_init:
                best_i = np.argmin(self._Y)
                l['incumbents'], _, track = self._create_param_dict((self._X[best_i][:-1], 1), {})
                l['incumbents_estimated_performance'] = -1
                l['incumbents_log'] = track['config_log']
            else:
                inc, inc_val = self._projected_incumbent_estimation(self._model_objective, self._X[:, :-1])
                l['incumbents'], _, track = self._create_param_dict((inc[:-1], 1), {})
                l['incumbents_estimated_performance'] = inc_val
                l.update({'incumbent_time': time()-start})

        l.update(tracking_vars)

        with open(os.path.join(
                self._log['path'],
                self._log['bn']+'_it{it}.json'.format(it=self._it)
        ), 'w') as f:
            json.dump(l, f)

    def _transform(self, s):
        '''
        Tranforms the subset-fragmentation to log-space

        :param s: subset-fragmentation
        :type s: int

        :return: Tranformed fragmentation
        :rtype: float
        '''
        s_transform = (np.log2(s) - np.log2(self._s_min)) \
                      / (np.log2(self._s_max) - np.log2(self._s_min))
        return s_transform

    def _retransform(self, s_transform):
        '''
        Retransforms the subset-fragmentation from log-space to normal space

        :param s_transform: transformed log-space s
        :type s_transform: float

        :return: normal-space s
        :rtype: int
        '''
        s = np.rint(2**(s_transform * (np.log2(self._s_max) \
                                       - np.log2(self._s_min)) \
                        + np.log2(self._s_min)))
        return int(s)

    def _init_random_uniform(self, lower, upper, n_points):
        '''
        Samples N data points uniformly.

        :param lower: Lower bounds of the input space
        :type lower: np.ndarray(D)
        :param upper: Upper bounds of the input space
        :type upper: np.ndarray(D)
        :param n_points: The number of initial data points
        :type n_points: int
        :param rng: Random number generator
        :type rng: np.random.RandomState

        :return: The initial design data points
        :rtype: np.ndarray(N,D)
        '''

        n_dims = lower.shape[0]

        return np.array([self._rng.uniform(lower, upper, n_dims) for _ in range(n_points)])
