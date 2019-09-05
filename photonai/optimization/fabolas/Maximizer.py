import sys
import os
import DIRECT
import numpy as np
import scipy
import emcee
import abc
from . import epmgp
from copy import deepcopy
from scipy.stats import norm

from multiprocessing import Pool, cpu_count
from functools import partial

from photonai.photonlogger import Logger


class BaseMaximizer(object):
    def __init__(self, objective_function, lower, upper, rng=None):
        """
        Interface for optimizers that maximizing the
        acquisition function.

        :param objective_function: The acquisition function which will be maximized
        :type objective_function: acquisition function
        :param lower: Lower bounds of the input space
        :type lower: np.ndarray (D)
        :param upper: Upper bounds of the input space
        :type upper: np.ndarray (D)
        :param rng: Random number generator
        :type rng: numpy.random.RandomState
        """
        self.lower = lower
        self.upper = upper
        self.objective_func = objective_function
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(10000))
        else:
            self.rng = rng

    def maximize(self):
        pass


class Direct(BaseMaximizer):
    def __init__(self, objective_function, lower, upper, logfilename="maximizer_log.txt",
                 n_func_evals=400, n_iters=200, verbose=False):
        """
        Interface for the DIRECT algorithm by D. R. Jones, C. D. Perttunen
        and B. E. Stuckmann

        :param objective_function: The acquisition function which will be maximized
        :type objective_function: acquisition function
        :param lower: Lower bounds of the input space
        :type lower: np.ndarray (D)
        :param upper: Upper bounds of the input space
        :type upper: np.ndarray (D)
        :param logfilename: File to store Direct's logs
        :type logfilename: string
        :param n_func_evals: The maximum number of function evaluations
        :type n_func_evals: int
        :param n_iters: The maximum number of iterations
        :type n_iters: int
        :param verbose: Suppress Direct's output.
        :type verbose: bool
        """
        self.n_func_evals = n_func_evals
        self.n_iters = n_iters
        self.verbose = verbose
        self.logfilename = logfilename

        super(Direct, self).__init__(objective_function, lower, upper)

    def _direct_acquisition_fkt_wrapper(self, acq_f):
        def _l(x, user_data):
            return -acq_f(np.array([x])), 0

        return _l

    def maximize(self):
        """
        Maximizes the given acquisition function.

        :return: Point with highest acquisition value.
        :rtype: np.ndarray(N,D)
        """
        if self.verbose:
            x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(self.objective_func),
                                   l=[self.lower],
                                   u=[self.upper],
                                   maxT=self.n_iters,
                                   maxf=self.n_func_evals,
                                   logfilename=self.logfilename)
        else:
            fileno = sys.stdout.fileno()
            with os.fdopen(os.dup(fileno), 'wb') as stdout:
                with os.fdopen(os.open(os.devnull, os.O_WRONLY), 'wb') as devnull:
                    sys.stdout.flush();
                    os.dup2(devnull.fileno(), fileno)  # redirect
                    x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(self.objective_func),
                                           l=[self.lower],
                                           u=[self.upper],
                                           maxT=self.n_iters,
                                           maxf=self.n_func_evals,
                                           logfilename=self.logfilename)
                sys.stdout.flush();
                os.dup2(stdout.fileno(), fileno)  # restore
        return x


class BaseAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        """
        A base class for acquisition_functions functions.

        :param model: Models the objective function.
        :type model: Model object
        """
        self.model = model

    def update(self, model):
        """
        This method will be called if the model is updated. E.g.
        Entropy search uses it to update it's approximation of P(x=x_min)

        :param model: Models the objective function.
        :type model: Model object
        """

        self.model = model

    def _multiple_inputs(foo):
        def wrapper(self, X, **kwargs):
            if len(X.shape) > 1:
                a = [foo(self, x, **kwargs) for x in X]
            else:
                a = foo(self, X, **kwargs)
            return a

        return wrapper

    @abc.abstractmethod
    def compute(self, x, derivative=False):
        """
        Computes the acquisition_functions value for a given point X. This function has
        to be overwritten in a derived class.

        :param x: The input point where the acquisition_functions function
            should be evaluate.
        :type x: np.ndarray(D,)

        :param derivative: If is set to true also the derivative of the acquisition_functions
            function at X is returned
        :type derivative: Boolean
        """
        pass

    def __call__(self, x, **kwargs):
        return self.compute(x, **kwargs)

    def get_json_data(self):
        """
        Json getter function

        :return:
        :rtype: Dict() object
        """

        json_data = dict()
        json_data = {"type": __name__}
        return json_data


class LogEI(BaseAcquisitionFunction):
    def __init__(self, model, par=0.0, **kwargs):

        r"""
        Computes for a given x the logarithm expected improvement as
        acquisition_functions value.

        :param model:
            A model that implements at least
                 - predict(X)
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)
        :type model: Model object
        :param par: Controls the balance between exploration
            and exploitation of the acquisition_functions function. Default is 0.01
        :type par: float
        """

        super(LogEI, self).__init__(model)

        self.par = par

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the Log EI value and its derivatives.

        :param X: The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        :type X: np.ndarray(1, D)
        :param derivative: If is set to true also the derivative of the acquisition_functions
            function at X is returned
            Not implemented yet!
        :type derivative: Boolean

        :return: Log Expected Improvement of X, Derivative of Log Expected Improvement at X (only if derivative=True)
        :rtype: np.ndarray(1,1), np.ndarray(1,D)
        """
        if derivative:
            raise Exception("LogEI does not support derivative \
                calculation until now")
            return

        m, v = self.model.predict(X)

        _, eta = self.model.get_incumbent()

        f_min = eta - self.par

        s = np.sqrt(v)

        z = (f_min - m) / s

        log_ei = np.zeros([m.size])
        for i in range(0, m.size):
            mu, sigma = m[i], s[i]

            #    par_s = self.par * sigma

            # Degenerate case 1: first term vanishes
            if np.any(abs(f_min - mu) == 0):
                if sigma > 0:
                    log_ei[i] = np.log(sigma) + norm.logpdf(z[i])
                else:
                    log_ei[i] = -np.Infinity
            # Degenerate case 2: second term vanishes and first term
            # has a special form.
            elif sigma == 0:
                if np.any(mu < f_min):
                    log_ei[i] = np.log(f_min - mu)
                else:
                    log_ei[i] = -np.Infinity
            # Normal case
            else:
                b = np.log(sigma) + norm.logpdf(z[i])
                # log(y+z) is tricky, we distinguish two cases:
                if np.any(f_min > mu):
                    # When y>0, z>0, we define a=ln(y), b=ln(z).
                    # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                    # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                    a = np.log(f_min - mu) + norm.logcdf(z[i])

                    log_ei[i] = max(a, b) + np.log(1 + np.exp(-abs(b - a)))
                else:
                    # When y<0, z>0, we define a=ln(-y), b=ln(z),
                    # and it has to be true that b >= a in
                    # order to satisfy y+z>=0.
                    # Then y+z = exp[ b + ln(exp(b-a) -1) ],
                    # and thus log(y+z) = a + ln(exp(b-a) -1)
                    a = np.log(mu - f_min) + norm.logcdf(z[i])
                    if a >= b:
                        # a>b can only happen due to numerical inaccuracies
                        # or approximation errors
                        log_ei[i] = -np.Infinity
                    else:
                        log_ei[i] = b + np.log(1 - np.exp(a - b))

        return log_ei


class InformationGain(BaseAcquisitionFunction):
    def __init__(self, model, lower, upper,
                 Nb=50, Np=400, sampling_acquisition=None,
                 sampling_acquisition_kw={"par": 0.0},
                 rng=None, verbose=False, **kwargs):

        """
        The Information Gain acquisition_functions function for Entropy Search [1].
        In a nutshell entropy search approximates the
        distribution pmin of the global minimum and tries to decrease its
        entropy. See Hennig and Schuler[1] for a more detailed view.

        [1] Hennig and C. J. Schuler
            Entropy search for information-efficient global optimization
            Journal of Machine Learning Research, 13, 2012

        :param model:
            A model that implements at least
                 - predict(X)
                 - predict_variances(X1, X2)
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)
        :type model: Model object
        :param lower: Lower bounds of the input space
        :type lower: np.ndarray (D)
        :param upper: Upper bounds of the input space
        :type upper: np.ndarray (D)
        :param Nb: Number of representer points to define pmin.
        :type Nb: int
        :param Np: Number of hallucinated values to compute the innovations
            at the representer points
        :type Np: int
        :param sampling_acquisition: Proposal measurement from which the representer points will
            be samples
        :type sampling_acquisition: function
        :param sampling_acquisition_kw: Additional keyword parameters that are passed to the
            acquisition_functions function
        :type sampling_acquisition_kw: dict
        """

        self.Nb = Nb
        super(InformationGain, self).__init__(model)
        self.lower = lower
        self.upper = upper
        self.D = self.lower.shape[0]
        self.sn2 = None

        if sampling_acquisition is None:
            sampling_acquisition = LogEI
        self.sampling_acquisition = sampling_acquisition(
            model, **sampling_acquisition_kw)

        self.Np = Np

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

    def loss_function(self, logP, lmb, lPred, *args):

        H = -np.sum(np.multiply(np.exp(logP), (logP + lmb)))  # current entropy
        dHp = - np.sum(np.multiply(np.exp(lPred),
                                   np.add(lPred, lmb)), axis=0) - H
        return np.array([dHp])

    def compute(self, X_test, derivative=False, **kwargs):
        """
        Computes the information gain of X and its derivatives

        :param X_test: The input point where the acquisition_functions function
            should be evaluate.
        :type X_test: np.ndarray(N, D)
        :param derivative: If is set to true also the derivative of the acquisition_functions
            function at X is returned
            Not tested!
        :type derivative: Boolean

        :return: Relative change of entropy of pmin, Derivatives with respect to X (only if derivative=True)
        :rtype: np.ndarray(N,), np.ndarray(N,D)
        """

        acq = np.zeros([X_test.shape[0]])
        grad = np.zeros([X_test.shape[0], X_test.shape[1]])
        for i, X in enumerate(X_test):
            if derivative:
                acq[i], grad[i] = self.dh_fun(X[None, :], derivative=True)

            else:
                acq[i] = self.dh_fun(X[None, :], derivative=False)

            if np.any(np.isnan(acq[i])) or np.any(acq[i] == np.inf):
                acq[i] = -sys.float_info.max

        if derivative:
            return acq, grad
        else:
            return acq

    def sampling_acquisition_wrapper(self, x):
        if np.any(x < self.lower) or np.any(x > self.upper):
            return -np.inf
        return self.sampling_acquisition(np.array([x]))[0]

    def sample_representer_points(self):
        self.sampling_acquisition.update(self.model)

        for i in range(5):
            restarts = np.zeros((self.Nb, self.D))
            restarts[0:self.Nb, ] = self.lower + (self.upper - self.lower) \
                                                 * self.rng.uniform(size=(self.Nb, self.D))
            sampler = emcee.EnsembleSampler(
                self.Nb, self.D, self.sampling_acquisition_wrapper)
            # zb are the representer points and lmb are their log EI values
            self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 50)
            if not np.any(np.isinf(self.lmb)):
                break
            else:
                if self.verbose:
                    Logger().debug("Fabolas.InformationGain: Infinity")

        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

    def update(self, model):
        self.model = model

        self.sn2 = self.model.get_noise()
        self.sample_representer_points()
        mu, var = self.model.predict(np.array(self.zb), full_cov=True)

        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = \
            epmgp.joint_min(mu, var, with_derivatives=True)

        self.W = scipy.stats.norm.ppf(np.linspace(1. / (self.Np + 1),
                                                  1 - 1. / (self.Np + 1),
                                                  self.Np))[np.newaxis, :]

        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def _dh_fun(self, x):
        # Number of belief locations:
        N = self.logP.size

        # Evaluate innovation
        dMdx, dVdx = self.innovations(x, self.zb)
        # The transpose operator is there to make the array indexing equivalent
        # to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMM = dMdx.dot(dMdx.T)
        trterm = np.sum(np.sum(np.multiply(self.dlogPdMudMu, np.reshape(
            dMM, (1, dMM.shape[0], dMM.shape[1]))), 2), 1)[
                 :, np.newaxis]

        # add a second dimension to the arrays if necessary:
        logP = np.reshape(self.logP, (self.logP.shape[0], 1))

        # Deterministic part of change:
        detchange = self.dlogPdSigma.dot(dVdx) + 0.5 * trterm
        # Stochastic part of change:
        stochange = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        lPred = np.add(logP + detchange, stochange)
        _maxLPred = np.amax(lPred, axis=0)
        s = _maxLPred + np.log(np.sum(np.exp(lPred - _maxLPred), axis=0))
        lselP = _maxLPred if np.any(np.isinf(s)) else s

        # Normalise:
        lPred = np.subtract(lPred, lselP)

        # We maximize the information gain
        dHp = -self.loss_function(logP, self.lmb, lPred, self.zb)
        dH = np.mean(dHp)
        return dH

    def dh_fun(self, x, derivative=False):

        if not (np.all(np.isfinite(self.lmb))):
            raise ValueError("lmb should not be infinite.")

        D = x.shape[1]
        # If x is a vector, convert it to a matrix (some functions are
        # sensitive to this distinction)
        if len(x.shape) == 1:
            x = x[np.newaxis]

        if np.any(x < self.lower) or np.any(x > self.upper):
            dH = np.spacing(1)
            ddHdx = np.zeros((x.shape[1], 1))
            return np.array([[dH]]), np.array([[ddHdx]])

        dH = self._dh_fun(x)

        if not np.isreal(dH):
            raise Exception("dH is not real")
        # Numerical derivative, renormalisation makes analytical derivatives
        # unstable.
        e = 1.0e-5
        if derivative:
            ddHdx = np.zeros((1, D))
            for d in range(D):
                # ## First part:
                y = np.array(x)
                y[0, d] += e
                dHy1 = self._dh_fun(y)
                # ## Second part:
                y = np.array(x)
                y[0, d] -= e
                dHy2 = self._dh_fun(y)

                ddHdx[0, d] = np.divide((dHy1 - dHy2), 2 * e)
                ddHdx = -ddHdx
            # endfor
            if len(ddHdx.shape) == 3:
                return_df = ddHdx
            else:
                return_df = np.array([ddHdx])
            return np.array([[dH]]), return_df
        return np.array([dH])

    def innovations(self, x, rep):
        # Get the variance at x with noise
        _, v = self.model.predict(x)
        v = v.reshape(-1, 1)

        # Get the variance at x without noise
        v_ = v - self.sn2

        # Compute the variance between the test point x and
        # the representer points
        sigma_x_rep = self.model.predict_variance(rep, x)

        norm_cov = np.dot(sigma_x_rep, np.linalg.inv(v_))
        # Compute the stochastic innovation for the mean
        dm_rep = np.dot(norm_cov, np.linalg.cholesky(v + 1e-10))

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep


class InformationGainPerUnitCost(InformationGain):
    def __init__(self, model, cost_model,
                 lower, upper,
                 is_env_variable,
                 n_representer=50,
                 verbose=False):
        """
        Information gain per unit cost as described in Swersky et al. [1] which
        computes the information gain of a configuration divided by it's cost.

        This implementation slightly differs from the implementation of
        Swersky et al. as it additionally adds the optimization overhead to
        the cost. You can simply set the optimization overhead to 0 to obtain
        the original formulation.

        [1] Swersky, K., Snoek, J., and Adams, R.
            Multi-task Bayesian optimization.
            In Proc. of NIPS 13, 2013.

        :param model: Models the objective function. The model has to be a
            Gaussian process.
        :type model: Model object
        :param cost_model: Models the cost function. The model has to be a Gaussian Process.
        :type cost_model: model
        :param lower: Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        :type lower: numpy array (D)
        :param upper : Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        :type upper: numpy array (D)
        :param is_env_variable: Specifies which input dimension is an environmental variable. If
            the i-th input is an environmental variable than the i-th entry has
            to be 1 and 0 otherwise.
        :type is_env_variable: numpy array (D)
        :param n_representer: The number of representer points to discretize the input space and
            to compute pmin.
        :type n_representer: int
        :param verbose: Print hyperparameter-configurations if true
        :type verbose: bool
        """
        self.cost_model = cost_model
        self.n_dims = lower.shape[0]
        self.verbose=verbose

        self.is_env = is_env_variable

        super(InformationGainPerUnitCost, self).__init__(model,
                                                         lower,
                                                         upper,
                                                         Nb=n_representer)

    def update(self, model, cost_model, overhead=None):
        self.cost_model = cost_model
        if overhead is None:
            self.overhead = 0
        else:
            self.overhead = overhead
        super(InformationGainPerUnitCost, self).update(model)

    def compute(self, X, derivative=False):
        """
        Computes the acquisition_functions value for a single point.

        :param X: The input point for which the acquisition_functions functions is computed.
        :type X: numpy array (1, D)
        :param derivative: If it is equal to True also the derivatives with respect to X is
            computed.
        :type derivative: bool


        :return: acquisition_value, grad
            acquisition_value: The acquisition_functions value computed for X.
            grad: The computed gradient of the acquisition_functions function at X. Only
                returned if derivative==True
        :rtype: numpy array
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        # Predict the log costs for this configuration
        log_cost = self.cost_model.predict(X)[0]

        if derivative:
            raise "Not implemented"
        else:
            dh = super(InformationGainPerUnitCost, self).compute(X,
                                                                 derivative=derivative)
            # We model the log cost, but we compute
            # the information gain per unit cost

            # Add the cost it took to pick the last configuration
            cost = np.exp(log_cost)

            acquisition_value = dh / (cost + self.overhead)

            return acquisition_value

    def sampling_acquisition_wrapper(self, x):

        # Check if sample point is inside the configuration space
        lower = self.lower[np.where(self.is_env == 0)]
        upper = self.upper[np.where(self.is_env == 0)]
        if np.any(x < lower) or np.any(x > upper):
            return -np.inf

        # Project point to subspace
        proj_x = np.concatenate((x, self.upper[self.is_env == 1]))
        return self.sampling_acquisition(np.array([proj_x]))[0]

    def sample_representer_points(self):
        # Sample representer points only in the
        # configuration space by setting all environmental
        # variables to 1
        D = np.where(self.is_env == 0)[0].shape[0]

        lower = self.lower[np.where(self.is_env == 0)]
        upper = self.upper[np.where(self.is_env == 0)]

        self.sampling_acquisition.update(self.model)

        for i in range(5):
            restarts = np.random.uniform(low=lower,
                                         high=upper,
                                         size=(self.Nb, D))
            sampler = emcee.EnsembleSampler(self.Nb, D,
                                            self.sampling_acquisition_wrapper)

            self.zb, self.lmb, _ = sampler.run_mcmc(restarts, 50)
            if not np.any(np.isinf(self.lmb)):
                break
            else:
                if self.verbose:
                    Logger().debug("Fabolas.InformationGainPerUnitCost: Infinity")
        if np.any(np.isinf(self.lmb)):
            raise ValueError("Could not sample valid representer points! LogEI is -infinity")
        if len(self.zb.shape) == 1:
            self.zb = self.zb[:, None]
        if len(self.lmb.shape) == 1:
            self.lmb = self.lmb[:, None]

        # Project representer points to subspace
        proj = np.ones([self.zb.shape[0],
                        self.upper[self.is_env == 1].shape[0]])
        proj *= self.upper[self.is_env == 1].shape[0]
        self.zb = np.concatenate((self.zb, proj), axis=1)


class MarginalizationGPMCMC(BaseAcquisitionFunction):
    _pool = None
    _pool_use_count = 0

    def __init__(self, acquisition_func, pool_size=-1):
        """
        Meta acquisition_functions function that allows to marginalise the
        acquisition_functions function over GP hyperparameters.

        :param acquisition_func: The acquisition_functions function that will be integrated.
        :type acquisition_func: BaseAcquisitionFunction object
        :param pool_size: Thread count to use for calculations of all MarginalizationGPMCMC-instances. Automatic calculation for <0 or None
        :type pool_size: int
        """
        if pool_size is not None and pool_size < 0:
            pool_size = min(len(acquisition_func.model.models), cpu_count())
            pool_size = max(pool_size, 1)
        self.acquisition_func = acquisition_func
        self.model = acquisition_func.model
        self.pool = pool_size

        # Save also the cost model if the acquisition_functions function needs it
        if hasattr(acquisition_func, "cost_model"):
            self.cost_model = acquisition_func.cost_model
        else:
            self.cost_model = None

        self.estimators = self.get_estimators(acquisition_func)

    @property
    def pool(self):
        return self.__class__._pool

    @pool.setter
    def pool(self, pool_size):
        cls = self.__class__
        if cls._pool is None:
            if pool_size is not None:
                pool_size = int(pool_size)
                if pool_size < 1:
                    pool_size = None
            cls._pool = Pool(pool_size)
        cls._pool_use_count += 1

    @pool.deleter
    def pool(self):
        cls = self.__class__
        cls._pool_use_count -= 1
        if cls._pool_use_count < 1:
            self._pool.close()
            self._pool.join()

    def get_estimators(self, acquisition_func):
        if len(self.model.models) == 0:
            return []

        # Keep for each model an extra acquisition_functions function module
        model_args = []
        for i in range(len(self.model.models)):
            if self.cost_model is not None:
                if len(self.cost_model.models) == 0:
                    model_args.append(None)
                else:
                    model_args.append(self.cost_model.models[i])
            else:
                model_args.append(self.model.models[i])

        return self.pool.map(
            partial(self.generate_estimator, acquisition_func=acquisition_func),
            model_args
        )

    @staticmethod
    def generate_estimator(model, acquisition_func):
        estimator = deepcopy(acquisition_func)
        estimator.model = model
        return estimator

    def update(self, model, cost_model=None, **kwargs):
        """
        Updates each acquisition_functions function object if the models
        have changed

        :param model: The model of the objective function, it has to be an instance of
            GaussianProcessMCMC.
        :type model: Model object
        :param cost_model: If the acquisition_functions function also takes the cost into account, we
            have to specify here the model for the cost function. cost_model
            has to be an instance of GaussianProcessMCMC.
        :type cost_model: Model object
        """
        if len(self.estimators) == 0:
            self.estimators = self.get_estimators(self.acquisition_func)

        self.model = model
        if cost_model is not None:
            self.cost_model = cost_model

        self.estimators = self.pool.starmap(
            partial(self.update_estimator, kw=kwargs),
            zip(
                self.estimators,
                self.model.models,
                self.cost_model.models if cost_model is not None\
                    else [None]*len(self.model.models)
            )
        )

    @staticmethod
    def update_estimator(estimator, mdl, mdl_cost, kw):
        if mdl_cost is not None:
            estimator.update(mdl, mdl_cost, **kw)
        else:
            estimator.update(mdl, **kw)
        return estimator

    def compute(self, X_test, derivative=False):
        """
        Integrates the acquisition_functions function over the GP's hyperparameters by
        averaging the acquisition_functions value for X of each hyperparameter sample.

        :param X_test: The input point where the acquisition_functions function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        :type X_test: np.ndarray(N, D)

        :param derivative: If is set to true also the derivative of the acquisition_functions
            function at X is returned
        :type derivative: Boolean

        :returns: Integrated acquisition_functions value of X, Derivative of the acquisition_functions value at X (only if derivative=True)
        :rtype: np.ndarray(1,1), np.ndarray(1,D)
        """
        acquisition_values = np.zeros([len(self.model.models), X_test.shape[0]])

        # Integrate over the acquisition_functions values
        acquisition_values[:len(self.model.models)] = self.pool.map(
            partial(self._integrate_acquisition_values, derivative=derivative, x=X_test),
            self.estimators[:len(self.model.models)]
        )

        return acquisition_values.mean(axis=0)

    @staticmethod
    def _integrate_acquisition_values(estimator, x, derivative):
        return estimator.compute(x, derivative=derivative)
