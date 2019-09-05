import abc
import numpy as np
import george
import emcee
from scipy import optimize

from multiprocessing import Pool, cpu_count
from functools import partial

from copy import deepcopy
from . import normalization

from photonai.photonlogger import Logger


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Abstract base class for all models
        """
        self.X = None
        self.y = None

    @abc.abstractmethod
    def train(self, X, y):
        """
        Trains the model on the provided data.

        :param X: Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        :type X: np.ndarray (N, D)
        :param y: The corresponding target values of the input data points.
        :type y: np.ndarray (N,)
        """
        pass

    def update(self, X, y):
        """
        Update the model with the new additional data. Override this function if your
        model allows to do something smarter than simple retraining

        :param X: Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        :type X: np.ndarray (N, D)
        :param y: The corresponding target values of the input data points.
        :type y: np.ndarray (N,)
        """
        X = np.append(self.X, X, axis=0)
        y = np.append(self.y, y, axis=0)
        self.train(X, y)

    @abc.abstractmethod
    def predict(self, X_test):
        """
        Predicts for a given set of test data points the mean and variance of its target values

        :param X_test: N Test data points with input dimensions D
        :type X_test: np.ndarray (N, D)

        :return: mean, var
            mean:
                Predictive mean of the test data points
            var:
                Predictive variance of the test data points
        :rtype: ndarray (N,), ndarray (N,)
        """
        pass

    def _check_shapes_train(func):
        def func_wrapper(self, X, y, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)

        return func_wrapper

    def _check_shapes_predict(func):
        def func_wrapper(self, X, *args, **kwargs):
            assert len(X.shape) == 2
            return func(self, X, *args, **kwargs)

        return func_wrapper

    def get_json_data(self):
        """
        Json getter function'

        :return:
        :rtype: dictionary
        """
        json_data = {'X': self.X if self.X is None else self.X.tolist(),
                     'y': self.y if self.y is None else self.y.tolist(),
                     'hyperparameters': ""}
        return json_data

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        :return: incumbent, incumbent_value
            incumbent:
                current incumbent
            incumbent_value:
                the observed value of the incumbent
        :rtype: ndarray (D,), ndarray (N,)
        """
        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]


class GaussianProcessMCMC(BaseModel):
    _pool = None
    _pool_use_count = 0

    def __init__(self, kernel, prior=None, n_hypers=20, chain_length=2000, burnin_steps=2000,
                 normalize_output=False, normalize_input=True, pool_size=-1,
                 rng=None, lower=None, upper=None, noise=-8, verbose_gp=False):
        """
        GaussianProcess model based on the george GP library that uses MCMC
        sampling to marginalise over the hyperparmeters. If you use this class
        make sure that you also use the IntegratedAcqusition function to
        integrate over the GP's hyperparameter as proposed by Snoek et al.

        :param kernel: Specifies the kernel that is used for all Gaussian Process
        :type kernel: george kernel object
        :param prior: Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface. During MCMC sampling the
            lnlikelihood is multiplied with the prior.
        :type prior: prior object
        :param n_hypers: The number of hyperparameter samples. This also determines the
            number of walker for MCMC sampling as each walker will
            return one hyperparameter sample.
        :type n_hypers: int
        :param chain_length: The length of the MCMC chain. We start n_hypers walker for
            chain_length elements and we use the last sample
            in the chain as a hyperparameter sample.
        :type chain_length: int
        :param lower: Lower bound of the input space which is used for the input space normalization
        :type lower: np.array(D,)
        :param upper: Upper bound of the input space which is used for the input space normalization
        :type upper: np.array(D,)
        :param burnin_steps: The number of burnin elements before the actual MCMC sampling starts.
        :type burnin_steps: int
        :param pool_size: Thread count to use for calculations of all MarginalizationGPMCMC-instances. Automatic calculation for <0 or None
        :type pool_size: int
        :param rng: Random number generator
        :type rng: np.random.RandomState
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.verbose_gp = verbose_gp

        if pool_size < 0:
            pool_size = min(n_hypers, cpu_count())
            pool_size = max(pool_size, 1)
        self.pool = pool_size

        self.kernel = kernel
        self.prior = prior
        self.noise = noise
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.models = []
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.is_trained = False

        self.lower = lower
        self.upper = upper

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
        if cls._pool_use_count == 0:
            self._pool.close()
            self._pool.join()

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True, **kwargs):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and y

        :param X: Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        :type X: np.ndarray (N, D)
        :param y: The corresponding target values.
        :type y: np.ndarray (N,)
        :param do_optimize: If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        :type do_optimize: boolean
        """

        if self.normalize_input:
            # Normalize input to be in [0, 1]
            self.X, self.lower, self.upper = normalization.zero_one_normalization(X, self.lower, self.upper)

        else:
            self.X = X

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
            if self.y_std == 0:
                raise ValueError("Cannot normalize output. All targets have the same value")
        else:
            self.y = y

        # Use the mean of the data as mean for the GP
        self.mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=self.mean)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            len(self.kernel.pars) + 1,
                                            self.loglikelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = np.random.rand(self.n_hypers, len(self.kernel.pars) + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        else:
            self.hypers = self.gp.kernel[:].tolist()
            self.hypers.append(self.noise)
            self.hypers = [self.hypers]

        self.models = self.pool.map(
            partial(self._instantiate_GP, X=X, y=y, kernel=self.kernel,
                kwargs={
                    'normalize_output': self.normalize_output,
                    'normalize_input': self.normalize_input,
                    'lower': self.lower,
                    'upper': self.upper,
                    'rng': self.rng,
                    'verbose': self.verbose_gp
                }),
            self.hypers
        )

        self.is_trained = True

    @staticmethod
    def _instantiate_GP(sample, X, y, kernel, kwargs):
        kern = deepcopy(kernel)
        kern.pars = np.exp(sample[:-1])
        noise = np.exp(sample[-1])
        model = GaussianProcess(kernel, noise=noise, **kwargs)
        model.train(X, y, do_optimize=False)
        return model

    def loglikelihood(self, theta):
        """
        Return the loglikelihood (+ the prior) for a hyperparameter
        configuration theta.

        :param theta: Hyperparameter vector. Note that all hyperparameter are
            on a log scale.
        :type theta: np.ndarray(H)

        :return: lnlikelihood + prior
        :rtype: float
        """

        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if np.any((-20 > theta) + (theta > 20)):
            return -np.inf

        # The last entry is always the noise
        sigma_2 = np.exp(theta[-1])
        # Update the kernel and compute the lnlikelihood.
        self.gp.kernel.pars = np.exp(theta[:-1])

        try:
            self.gp.compute(self.X, yerr=np.sqrt(sigma_2))
        except:
            return -np.inf

        if self.prior is not None:
            return self.prior.lnprob(theta) + self.gp.lnlikelihood(self.y, quiet=True)
        else:
            return self.gp.lnlikelihood(self.y, quiet=True)

    @BaseModel._check_shapes_predict
    def predict(self, X_test, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function
        at X average over all hyperparameter samples.
        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)
        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2

        :param X_test: Input test points
        :type X_test: np.ndarray (N, D)

        :return: predictive mean, predictive variance
        :rtype: np.array(N,), np.array(N,)

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')
        #
        # if self.normalize_input:
        #     X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        # else:
        #     X_test_norm = X_test

        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        results = self.pool.map(partial(self._model_predict, x=X_test), self.models)
        for i, r in enumerate(results):
            mu[i], var[i] = r

        # See the Algorithm Runtime Prediction paper by Hutter et al.
        # for the derivation of the total variance
        m = mu.mean(axis=0)
        # v = np.mean(mu ** 2 + var) - m ** 2
        v = var.mean(axis=0)

        # if self.normalize_output:
        #     m = normalization.zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)
        #     v *= self.y_std ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v

    @staticmethod
    def _model_predict(model, x):
        return model.predict(x)

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        :return: incumbent, incumbent_value
            incumbent:
                current incumbent
            incumbent_value:
                the observed value of the incumbent
        :rtype: ndarray (D,), ndarray (N,)
        """
        inc, inc_value = super(GaussianProcessMCMC, self).get_incumbent()
        if self.normalize_input:
            inc = normalization.zero_one_unnormalization(inc, self.lower, self.upper)

        if self.normalize_output:
            inc_value = normalization.zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value


class GaussianProcess(BaseModel):
    def __init__(self, kernel, prior=None,
                 noise=1e-3, use_gradients=False,
                 normalize_output=False,
                 normalize_input=True,
                 lower=None, upper=None, rng=None, verbose=False):
        """
        Interface to the george GP library. The GP hyperparameter are obtained
        by optimizing the marginal log likelihood.

        :param kernel: Specifies the kernel that is used for all Gaussian Process
        :type kernel: george kernel object
        :param prior: Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface.
        :type prior: prior object
        :param noise: Noise term that is added to the diagonal of the covariance matrix
            for the Cholesky decomposition.
        :type noise: float
        :param use_gradients: Use gradient information to optimize the negative log likelihood
        :type use_gradients: bool
        :param lower: Lower bound of the input space which is used for the input space normalization
        :type lower: np.array(D,)
        :param upper: Upper bound of the input space which is used for the input space normalization
        :type upper: np.array(D,)
        :param normalize_output: Zero mean unit variance normalization of the output values
        :type normalize_output: bool
        :param normalize_input: Normalize all inputs to be in [0, 1]. This is important to define good priors for the
            length scales.
        :type normalize_input: bool
        :param rng: Random number generator
        :type rng: np.random.RandomState
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.verbose = verbose
        self.kernel = kernel
        self.gp = None
        self.prior = prior
        self.noise = noise
        self.use_gradients = use_gradients
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.hypers = []
        self.is_trained = False
        self.lower = lower
        self.upper = upper

    @BaseModel._check_shapes_train
    def train(self, X, y, do_optimize=True):
        """
        Computes the Cholesky decomposition of the covariance of X and
        estimates the GP hyperparameters by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of X.

        :param X: Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        :type X: np.ndarray (N, D)
        :param y: The corresponding target values.
        :type y: np.ndarray (N,)
        :param do_optimize: If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        :type do_optimize: boolean
        """

        if self.normalize_input:
            # Normalize input to be in [0, 1]
            self.X, self.lower, self.upper = normalization.zero_one_normalization(X, self.lower, self.upper)
        else:
            self.X = X

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
            if self.y_std == 0:
                raise ValueError("Cannot normalize output. All targets have the same value")
        else:
            self.y = y

        # Use the empirical mean of the data as mean for the GP
        self.mean = np.mean(self.y, axis=0)

        self.gp = george.GP(self.kernel, mean=self.mean)

        if do_optimize:
            self.hypers = self.optimize()
            self.gp.kernel[:] = self.hypers[:-1]
            self.noise = np.exp(self.hypers[-1])  # sigma^2
        else:
            self.hypers = self.gp.kernel[:]
            self.hypers = np.append(self.hypers, np.log(self.noise))

        if self.verbose:
            Logger().debug("Fabolas.GaussianProcess: GP Hyperparameters: " + str(self.hypers))

        self.gp.compute(self.X, yerr=np.sqrt(self.noise))

        self.is_trained = True

    def get_noise(self):
        return self.noise

    def nll(self, theta):
        """
        Returns the negative marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.
        (negative because we use scipy minimize for optimization)

        :param theta: Hyperparameter vector. Note that all hyperparameter are
            on a log scale.
        :type theta: np.ndarray(H)

        :return: lnlikelihood + prior
        :rtype: float
        """
        # Specify bounds to keep things sane
        if np.any((-20 > theta) + (theta > 20)):
            return 1e25

        # The last entry of theta is always the noise
        self.gp.kernel[:] = theta[:-1]
        noise = np.exp(theta[-1])  # sigma^2

        try:
            self.gp.compute(self.X, yerr=np.sqrt(noise))
        except np.linalg.LinAlgError:
            return 1e25

        ll = self.gp.lnlikelihood(self.y, quiet=True)

        # Add prior
        if self.prior is not None:
            ll += self.prior.lnprob(theta)

        # We add a minus here because scipy is minimizing
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(self, theta):

        self.gp.kernel[:] = theta[:-1]
        noise = np.exp(theta[-1])

        self.gp.compute(self.X, yerr=np.sqrt(noise))

        self.gp._compute_alpha(self.y)
        K_inv = self.gp.solver.apply_inverse(np.eye(self.gp._alpha.size),
                                             in_place=True)

        # The gradients of the Gram matrix, for the noise this is just
        # the identity matrix
        Kg = self.gp.kernel.gradient(self.gp._x)
        Kg = np.concatenate((Kg, np.eye(Kg.shape[0])[:, :, None]), axis=2)

        # Calculate the gradient.
        A = np.outer(self.gp._alpha, self.gp._alpha) - K_inv
        g = 0.5 * np.einsum('ijk,ij', Kg, A)

        if self.prior is not None:
            g += self.prior.gradient(theta)

        return -g

    def optimize(self):
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        :return: theta: Hyperparameter vector that maximizes the marginal log likelihood
        :rtype: np.ndarray(H)
        """
        # Start optimization from the previous hyperparameter configuration
        p0 = self.gp.kernel.vector
        p0 = np.append(p0, np.log(self.noise))

        if self.use_gradients:
            bounds = [(-10, 10)] * (len(self.kernel) + 1)
            theta, _, _ = optimize.fmin_l_bfgs_b(self.nll, p0,
                                                 fprime=self.grad_nll,
                                                 bounds=bounds)
        else:
            try:
                results = optimize.minimize(self.nll, p0)
                theta = results.x
            except ValueError:
                Logger().error("Fabolas.GaussianProcess: Could not find a valid hyperparameter configuration! Use initial configuration")
                theta = p0

        return theta

    def predict_variance(self, x1, X2):
        r"""
        Predicts the variance between a test points x1 and a set of points X2 by
           math: \sigma(X_1, X_2) = k_{X_1,X_2} - k_{X_1,X} * (K_{X,X}
                       + \sigma^2*\mathds{I})^-1 * k_{X,X_2})

        :param x1: First test point
        :type x1: np.ndarray (1, D)
        :param X2: Set of test point
        :type X2: np.ndarray (N, D)

        :return: predictive variance between x1 and X2
        :rtype: np.array(N, 1)
        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        # if self.normalize_input:
        #     x1_norm, _, _ = normalization.zero_one_normalization(x1, self.lower, self.upper)
        #     X2_norm, _, _ = normalization.zero_one_normalization(X2, self.lower, self.upper)
        # else:
        #     x1_norm = x1
        #     X2_norm = X2

        x_ = np.concatenate((x1, X2))
        _, var = self.predict(x_, full_cov=True)

        var = var[-1, :-1, np.newaxis]

        return var

    @BaseModel._check_shapes_predict
    def predict(self, X_test, full_cov=False, **kwargs):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        :param X_test: Input test points
        :type X: np.ndarray (N, D)
        :param full_cov: If set to true than the whole covariance matrix between the test points is returned
        :type fill_cov: bool

        :return: predictive mean, predictive variance
        :rtype: np.array(N,), np.array(N,) or np.array(N, N) if full_cov == True

        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        mu, var = self.gp.predict(self.y, X_test_norm)

        if self.normalize_output:
            mu = normalization.zero_mean_unit_var_unnormalization(mu, self.y_mean, self.y_std)
            var *= self.y_std ** 2
        if not full_cov:
            var = np.diag(var)

        # Clip negative variances and set them to the smallest
        # positive float value
        if var.shape[0] == 1:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
        else:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

        return mu, var

    def sample_functions(self, X_test, n_funcs=1):
        """
        Samples F function values from the current posterior at the N
        specified test points.

        :param X_test: Input test points
        :type X_test: np.ndarray (N, D)
        :param n_funcs: Number of function values that are drawn at each test point.
        :type n_funcs: int

        :return: function_samples: The F function values drawn at the N test points.
        :rtype: np.array(F, N)
        """

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        funcs = self.gp.sample_conditional(self.y, X_test_norm, n_funcs)

        if self.normalize_output:
            funcs = normalization.zero_mean_unit_var_unnormalization(funcs, self.y_mean, self.y_std)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        :return: current incumbent, the observed value of the incumbent
        :rtype: ndarray (D,), ndarray (N,)
        """
        inc, inc_value = super(GaussianProcess, self).get_incumbent()
        if self.normalize_input:
            inc = normalization.zero_one_unnormalization(inc, self.lower, self.upper)

        if self.normalize_output:
            inc_value = normalization.zero_mean_unit_var_unnormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value


class FabolasGPMCMC(GaussianProcessMCMC):
    def __init__(self, kernel, basis_func,
                 prior=None, n_hypers=20,
                 chain_length=2000, burnin_steps=2000,
                 normalize_output=False,
                 rng=None,
                 lower=None,
                 upper=None,
                 noise=-8,
                 pool_size=-1,
                 verbose_gp=False):

        self.basis_func = basis_func
        self.hypers = None
        super(FabolasGPMCMC, self).__init__(kernel, prior,
                                            n_hypers, chain_length,
                                            burnin_steps,
                                            normalize_output=normalize_output,
                                            normalize_input=False,
                                            rng=rng, lower=lower,
                                            upper=upper, noise=noise,
                                            pool_size=pool_size,
                                            verbose_gp=verbose_gp)

    def train(self, X, y, do_optimize=True, **kwargs):
        X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], self.lower, self.upper)
        s_ = self.basis_func(X[:, -1])[:, None]
        self.X = np.concatenate((X_norm, s_), axis=1)

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        # Use the mean of the data as mean for the GP
        mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=mean)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            len(self.kernel) + 1,
                                            self.loglikelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = np.random.rand(self.n_hypers, len(self.kernel.pars) + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        else:
            if self.hypers is None:
                self.hypers = self.gp.kernel[:].tolist()
                self.hypers.append(self.noise)
                self.hypers = [self.hypers]

        self.models = self.pool.map(
            partial(self._instantiate_GP, X=X, y=y, kernel=self.kernel,
                kw={
                    'basis_function': self.basis_func,
                    'normalize_output': self.normalize_output,
                    'lower': self.lower,
                    'upper': self.upper,
                    'rng': self.rng
                }),
            self.hypers
        )

        self.is_trained = True

    @staticmethod
    def _instantiate_GP(sample, X, y, kernel, kw):
        kern = deepcopy(kernel)
        kern.pars = np.exp(sample[:-1])
        noise = np.exp(sample[-1])
        model = FabolasGP(kern, noise=noise, **kw)
        model.train(X, y, do_optimize=False)
        return model


class FabolasGP(GaussianProcess):
    def __init__(self, kernel, basis_function, prior=None,
                 noise=1e-3, use_gradients=False,
                 normalize_output=False,
                 lower=None, upper=None, rng=None):
        self.basis_function = basis_function
        super(FabolasGP, self).__init__(kernel=kernel,
                                        prior=prior,
                                        noise=noise,
                                        use_gradients=use_gradients,
                                        normalize_output=normalize_output,
                                        normalize_input=False,
                                        lower=lower,
                                        upper=upper,
                                        rng=rng)

    def normalize(self, X):
        X_norm, _, _ = normalization.zero_one_normalization(X[:, :-1], self.lower, self.upper)
        s_ = self.basis_function(X[:, -1])[:, None]
        X_norm = np.concatenate((X_norm, s_), axis=1)
        return X_norm

    def train(self, X, y, do_optimize=True):
        self.original_X = X
        X_norm = self.normalize(X)
        return super(FabolasGP, self).train(X_norm, y, do_optimize)

    def predict(self, X_test, full_cov=False, **kwargs):
        X_norm = self.normalize(X_test)
        return super(FabolasGP, self).predict(X_norm, full_cov)

    def sample_functions(self, X_test, n_funcs=1):
        X_norm = self.normalize(X_test)
        return super(FabolasGP, self).sample_functions(X_norm, n_funcs)

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        :return: current incumbent, the observed value of the incumbent
        :rtype: ndarray (D,), ndarray (N,)

        """

        projection = np.ones([self.original_X.shape[0], 1]) * 1

        X_projected = np.concatenate((self.original_X[:, :-1], projection), axis=1)
        X_norm = self.normalize(X_projected)

        m, _ = self.predict(X_norm)

        best = np.argmin(m)
        incumbent = X_projected[best]
        incumbent_value = m[best]

        return incumbent, incumbent_value
