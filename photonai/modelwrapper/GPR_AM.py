from sklearn.base import BaseEstimator
class BaseModelWrapper(BaseEstimator):
    """
    The PHOTON interface for implementing custom pipeline elements.

    PHOTON works on top of the scikit-learn object API,
    [see documentation](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects)

    Your class should overwrite the following definitions:

    - `fit(data)`: learn or adjust to the data

    If it is an estimator, which means it has the ability to learn,

    - it should implement `predict(data)`: using the learned model to generate prediction
    - should inherit *sklearn.base.BaseEstimator* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html))
    - inherits *get_params* and *set_params*

    If it is an transformer, which means it preprocesses or prepares the data

    - it should implement `transform(data)`: applying the logic to the data to transform it
    - should inherit from *sklearn.base.TransformerMixin* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html))
    - inherits *fit_transform* as a concatenation of both fit and transform
    - should inherit *sklearn.base.BaseEstimator* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html))
    - inherits *get_params* and *set_params*

    `Prepare for hyperparameter optimization`

    PHOTON expects you to `define all parameters` that you want to optimize in the hyperparameter search in the
    `constructor stub`, and to be addressable with the `same name as class variable`.
    In this way you can define any parameter and it is automatically prepared for the hyperparameter search process.

    See the [scikit-learn object API documentation](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) for more in depth information about the interface.
    """

    def __init__(self, respfile=None, covfile=None, maskfile=None, cvfolds=None, testcov=None, testresp=None, saveoutput=True, outputsuffix=None):
        pass

    def fit(self, data, targets=None):
        """
        Adjust the underlying model or method to the data.

        Returns
        -------
        IMPORTANT: must return self!
        """

        """ Estimate a normative model

        This will estimate a model in one of two settings according to the
        particular parameters specified (see below):

        * under k-fold cross-validation
        * estimating a training dataset then applying to a second test dataset

        The models are estimated on the basis of data stored on disk in ascii or
        neuroimaging data formats (nifti or cifti). Ascii data should be in
        tab or space delimited format with the number of subjects in rows and the
        number of variables in columns. Neuroimaging data will be reshaped
        into the appropriate format

        Basic usage::

            estimate(respfile, covfile, [extra_arguments])

        where the variables are defined below. Note that either the cfolds
        parameter or (testcov, testresp) should be specified, but not both.

        :param respfile: response variables for the normative model
        :param covfile: covariates used to predict the response variable
        :param maskfile: mask used to apply to the data (nifti only)
        :param cvfolds: Number of cross-validation folds
        :param testcov: Test covariates
        :param testresp: Test responses
        :param saveoutput: Save the output to disk? Otherwise returned as arrays
        :param outputsuffix: Text string to add to the output filenames

        All outputs are written to disk in the same format as the input. These are:

        :outputs: * yhat - predictive mean
                  * ys2 - predictive variance
                  * Z - deviance scores
                  * Rho - Pearson correlation between true and predicted responses
                  * pRho - parametric p-value for this correlation
                  * rmse - root mean squared error between true/predicted responses
                  * smse - standardised mean squared error

        The outputsuffix may be useful to estimate multiple normative models in the
        same directory (e.g. for custom cross-validation schemes)
        """

        # load data
        print("Processing data in " + respfile)
        X = fileio.load(covfile)
        Y, maskvol = load_response_vars(respfile, maskfile)
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        Nmod = Y.shape[1]

        if testcov is not None:
            # we have a separate test dataset
            Xte = fileio.load(testcov)
            Yte, testmask = load_response_vars(testresp, maskfile)
            testids = range(X.shape[0], X.shape[0] + Xte.shape[0])

            if len(Yte.shape) == 1:
                Yte = Yte[:, np.newaxis]
            if len(Xte.shape) == 1:
                Xte = Xte[:, np.newaxis]

            # treat as a single train-test split
            splits = CustomCV((range(0, X.shape[0]),), (testids,))

            Y = np.concatenate((Y, Yte), axis=0)
            X = np.concatenate((X, Xte), axis=0)

            # force the number of cross-validation folds to 1
            if cvfolds is not None and cvfolds != 1:
                print("Ignoring cross-valdation specification (test data given)")
            cvfolds = 1
        else:
            # we are running under cross-validation
            splits = KFold(n_splits=cvfolds)
            testids = range(0, X.shape[0])

        # find and remove bad variables from the response variables
        # note: the covariates are assumed to have already been checked
        nz = np.where(np.bitwise_and(np.isfinite(Y).any(axis=0),
                                     np.var(Y, axis=0) != 0))[0]

        # starting hyperparameters. Could also do random restarts here
        covfunc = CovSum(X, ('CovLin', 'CovSqExpARD'))
        hyp0 = np.zeros(covfunc.get_n_params() + 1)

        # run cross-validation loop
        Yhat = np.zeros_like(Y)
        S2 = np.zeros_like(Y)
        Z = np.zeros_like(Y)
        nlZ = np.zeros((Nmod, cvfolds))
        Hyp = np.zeros((Nmod, len(hyp0), cvfolds))
        for idx in enumerate(splits.split(X)):
            fold = idx[0]
            tr = idx[1][0]
            te = idx[1][1]

            # standardize responses and covariates, ignoring invalid entries
            iy, jy = np.ix_(tr, nz)
            mY = np.mean(Y[iy, jy], axis=0)
            sY = np.std(Y[iy, jy], axis=0)
            Yz = np.zeros_like(Y)
            Yz[:, nz] = (Y[:, nz] - mY) / sY
            mX = np.mean(X[tr, :], axis=0)
            sX = np.std(X[tr, :], axis=0)
            Xz = (X - mX) / sX

            # estimate the models for all subjects
            for i in range(0, len(nz)):  # range(0, Nmod):
                print("Estimating model ", i + 1, "of", len(nz))
                gpr = GPR(hyp0, covfunc, Xz[tr, :], Yz[tr, nz[i]])
                Hyp[nz[i], :, fold] = gpr.estimate(hyp0, covfunc, Xz[tr, :],
                                                   Yz[tr, nz[i]])

                yhat, s2 = gpr.predict(Hyp[nz[i], :, fold], Xz[tr, :],
                                       Yz[tr, nz[i]], Xz[te, :])

                Yhat[te, nz[i]] = yhat * sY[i] + mY[i]
                S2[te, nz[i]] = np.diag(s2) * sY[i] ** 2
                Z[te, nz[i]] = (Y[te, nz[i]] - Yhat[te, nz[i]]) / \
                               np.sqrt(S2[te, nz[i]])
                nlZ[nz[i], fold] = gpr.nlZ

        # compute performance metrics
        MSE = np.mean((Y[testids, :] - Yhat[testids, :]) ** 2, axis=0)
        RMSE = np.sqrt(MSE)
        # for the remaining variables, we need to ignore zero variances
        SMSE = np.zeros_like(MSE)
        Rho = np.zeros(Nmod)
        pRho = np.ones(Nmod)
        iy, jy = np.ix_(testids, nz)  # ids for tested samples with nonzero values
        SMSE[nz] = MSE[nz] / np.var(Y[iy, jy], axis=0)
        Rho[nz], pRho[nz] = compute_pearsonr(Y[iy, jy], Yhat[iy, jy])

        # Set writing options
        if saveoutput:
            print("Writing output ...")
            if fileio.file_type(respfile) == 'cifti' or \
                    fileio.file_type(respfile) == 'nifti':
                exfile = respfile
            else:
                exfile = None
            if outputsuffix is not None:
                ext = str(outputsuffix) + fileio.file_extension(respfile)
            else:
                ext = fileio.file_extension(respfile)

            # Write output
            fileio.save(Yhat[testids, :].T, 'yhat' + ext,
                        example=exfile, mask=maskvol)
            fileio.save(S2[testids, :].T, 'ys2' + ext,
                        example=exfile, mask=maskvol)
            fileio.save(Z[testids, :].T, 'Z' + ext, example=exfile, mask=maskvol)
            fileio.save(Rho, 'Rho' + ext, example=exfile, mask=maskvol)
            fileio.save(pRho, 'pRho' + ext, example=exfile, mask=maskvol)
            fileio.save(RMSE, 'rmse' + ext, example=exfile, mask=maskvol)
            fileio.save(SMSE, 'smse' + ext, example=exfile, mask=maskvol)
            if cvfolds is None:
                fileio.save(Hyp, 'Hyp' + ext, example=exfile, mask=maskvol)
            else:
                for idx in enumerate(splits.split(X)):
                    fold = idx[0]
                    fileio.save(Hyp[:, :, fold], 'Hyp_' + str(fold + 1) +
                                ext, example=exfile, mask=maskvol)
        else:
            output = (Yhat, S2, Z, Rho, pRho, RMSE, SMSE)
            return output

    def predict(self, data):
        """
        Use the learned model to make predictions.
        """



# Core GP stuff
from __future__ import print_function
from __future__ import division
import numpy as np
from scipy import optimize
from numpy.linalg import solve, LinAlgError
from numpy.linalg import cholesky as chol
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from nispat.utils import squared_dist

# --------------------
# Covariance functions
# --------------------


class CovBase(with_metaclass(ABCMeta)):
    """ Base class for covariance functions.

        All covariance functions must define the following methods::

            CovFunction.get_n_params()
            CovFunction.cov()
            CovFunction.xcov()
            CovFunction.dcov()
    """

    def __init__(self, x=None):
        self.n_params = np.nan

    def get_n_params(self):
        """ Report the number of parameters required """

        assert not np.isnan(self.n_params), \
            "Covariance function not initialised"

        return self.n_params

    @abstractmethod
    def cov(self, theta, x, z=None):
        """ Return the full covariance (or cross-covariance if z is given) """

    @abstractmethod
    def dcov(self, theta, x, i):
        """ Return the derivative of the covariance function with respect to
            the i-th hyperparameter """


class CovLin(CovBase):
    """ Linear covariance function (no hyperparameters)
    """

    def __init__(self, x=None):
        self.n_params = 0
        self.first_call = False

    def cov(self, theta, x, z=None):
        if not self.first_call and not theta and theta is not None:
            self.first_call = True
            if len(theta) > 0 and theta[0] is not None:
                print("CovLin: ignoring unnecessary hyperparameter ...")

        if z is None:
            z = x

        K = x.dot(z.T)
        return K

    def dcov(self, theta, x, i):
        raise ValueError("Invalid covariance function parameter")


class CovSqExp(CovBase):
    """ Ordinary squared exponential covariance function.
        The hyperparameters are::

            theta = ( log(ell), log(sf2) )

        where ell is a lengthscale parameter and sf2 is the signal variance
    """

    def __init__(self, x=None):
        self.n_params = 2

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        if z is None:
            z = x

        R = squared_dist(x/self.ell, z/self.ell)
        K = self.sf2 * np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        self.ell = np.exp(theta[0])
        self.sf2 = np.exp(2*theta[1])

        R = squared_dist(x/self.ell, x/self.ell)

        if i == 0:   # return derivative of lengthscale parameter
            dK = self.sf2 * np.exp(-R/2) * R
            return dK
        elif i == 1:   # return derivative of signal variance parameter
            dK = 2*self.sf2 * np.exp(-R/2)
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")


class CovSqExpARD(CovBase):
    """ Squared exponential covariance function with ARD
        The hyperparameters are::

            theta = (log(ell_1, ..., log_ell_D), log(sf2))

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        self.D = x.shape[1]
        self.n_params = self.D + 1

    def cov(self, theta, x, z=None):
        self.ell = np.exp(theta[0:self.D])
        self.sf2 = np.exp(2*theta[self.D])

        if z is None:
            z = x

        R = squared_dist(x.dot(np.diag(1./self.ell)),
                         z.dot(np.diag(1./self.ell)))
        K = self.sf2*np.exp(-R/2)
        return K

    def dcov(self, theta, x, i):
        K = self.cov(theta, x)
        if i < self.D:    # return derivative of lengthscale parameter
            dK = K * squared_dist(x[:, i]/self.ell[i], x[:, i]/self.ell[i])
            return dK
        elif i == self.D:   # return derivative of signal variance parameter
            dK = 2*K
            return dK
        else:
            raise ValueError("Invalid covariance function parameter")


class CovSum(CovBase):
    """ Sum of covariance functions. These are passed in as a cell array and
        intialised automatically. For example::

            C = CovSum(x,(CovLin, CovSqExpARD))
            C = CovSum.cov(x, )

        The hyperparameters are::

            theta = ( log(ell_1, ..., log_ell_D), log(sf2) )

        where ell_i are lengthscale parameters and sf2 is the signal variance
    """

    def __init__(self, x=None, covfuncnames=None):
        if x is None:
            raise ValueError("N x D data matrix must be supplied as input")
        if covfuncnames is None:
            raise ValueError("A list of covariance functions is required")
        self.covfuncs = []
        self.n_params = 0
        for cname in covfuncnames:
            covfunc = eval(cname + '(x)')
            self.n_params += covfunc.get_n_params()
            self.covfuncs.append(covfunc)
        self.N, self.D = x.shape

    def cov(self, theta, x, z=None):
        theta_offset = 0
        for ci, covfunc in enumerate(self.covfuncs):
            n_params_c = covfunc.get_n_params()
            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if ci == 0:
                K = covfunc.cov(theta_c, x, z)
            else:
                K += covfunc.cov(theta_c, x, z)
        return K

    def dcov(self, theta, x, i):
        theta_offset = 0
        for covfunc in self.covfuncs:
            n_params_c = covfunc.get_n_params()
            theta_c = [theta[c] for c in
                       range(theta_offset, theta_offset + n_params_c)]
            theta_offset += n_params_c

            if theta_c:  # does the variable have any hyperparameters?
                if 'dK' not in locals():
                    dK = covfunc.dcov(theta_c, x, i)
                else:
                    dK += covfunc.dcov(theta_c, x, i)
        return dK

# -----------------------
# Gaussian process models
# -----------------------


class GPR:
    """Gaussian process regression

    Estimation and prediction of Gaussian process regression models

    Basic usage::

        G = GPR()
        hyp = B.estimate(hyp0, cov, X, y)
        ys, ys2 = B.predict(hyp, cov, X, y, Xs)

    where the variables are

    :param hyp: vector of hyperparmaters
    :param cov: covariance function
    :param X: N x D data array
    :param y: 1D Array of targets (length N)
    :param Xs: Nte x D array of test cases
    :param hyp0: starting estimates for hyperparameter optimisation

    :returns: * ys - predictive mean
              * ys2 - predictive variance

    The hyperparameters are::

        hyp = ( log(sn2), (cov function params) )  # hyp is a list or array

    The implementation and notation  follows Rasmussen and Williams (2006).
    As in the gpml toolbox, these parameters are estimated using conjugate
    gradient optimisation of the marginal likelihood. Note that there is no
    explicit mean function, thus the gpr routines are limited to modelling
    zero-mean processes.

    Reference:
    C. Rasmussen and C. Williams (2006) Gaussian Processes for Machine Learning

    Written by A. Marquand
    """

    def __init__(self, hyp=None, covfunc=None, X=None, y=None, n_iter=100,
                 tol=1e-3, verbose=False):

        self.hyp = np.nan
        self.nlZ = np.nan
        self.tol = tol          # not used at present
        self.n_iter = n_iter    # not used at present
        self.verbose = verbose

        if (hyp is not None) and (X is not None) and (y is not None):
            self.post(hyp, covfunc, X, y)

    def _updatepost(self, hyp, covfunc):

        hypeq = np.asarray(hyp == self.hyp)
        if hypeq.all() and hasattr(self, 'alpha') and \
           (hasattr(self, 'covfunc') and covfunc == self.covfunc):
            return False
        else:
            return True

    def post(self, hyp, covfunc, X, y):
        """ Generic function to compute posterior distribution.
        """

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        self.N, self.D = X.shape

        if not self._updatepost(hyp, covfunc):
            print("hyperparameters have not changed, using exising posterior")
            return

        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        if self.verbose:
            print("estimating posterior ... | hyp=", hyp)

        self.K = covfunc.cov(theta, X)
        self.L = chol(self.K + sn2*np.eye(self.N))
        self.alpha = solve(self.L.T, solve(self.L, y))
        self.hyp = hyp
        self.covfunc = covfunc

    def loglik(self, hyp, covfunc, X, y):
        """ Function to compute compute log (marginal) likelihood
        """

        # load or recompute posterior
        if self._updatepost(hyp, covfunc):
            try:
                self.post(hyp, covfunc, X, y)
            except (ValueError, LinAlgError):
                print("Warning: Estimation of posterior distribution failed")
                self.nlZ = 1/np.finfo(float).eps
                return self.nlZ

        self.nlZ = 0.5*y.T.dot(self.alpha) + sum(np.log(np.diag(self.L))) + \
                   0.5*self.N*np.log(2*np.pi)

        # make sure the output is finite to stop the minimizer getting upset
        if not np.isfinite(self.nlZ):
            self.nlZ = 1/np.finfo(float).eps

        if self.verbose:
            print("nlZ= ", self.nlZ, " | hyp=", hyp)

        return self.nlZ

    def dloglik(self, hyp, covfunc, X, y):
        """ Function to compute derivatives
        """

        # hyperparameters
        sn2 = np.exp(2*hyp[0])       # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        # load posterior and prior covariance
        if self._updatepost(hyp, covfunc):
            try:
                self.post(hyp, covfunc, X, y)
            except (ValueError, LinAlgError):
                print("Warning: Estimation of posterior distribution failed")
                dnlZ = np.sign(self.dnlZ) / np.finfo(float).eps
                return dnlZ

        # compute Q = alpha*alpha' - inv(K)
        Q = np.outer(self.alpha, self.alpha) - \
            solve(self.L.T, solve(self.L, np.eye(self.N)))

        # initialise derivatives
        self.dnlZ = np.zeros(len(hyp))

        # noise variance
        self.dnlZ[0] = -sn2*np.trace(Q)

        # covariance parameter(s)
        for par in range(0, len(theta)):
            # compute -0.5*trace(Q.dot(dK/d[theta_i])) efficiently
            dK = covfunc.dcov(theta, X, i=par)
            self.dnlZ[par+1] = -0.5*np.sum(np.sum(Q*dK.T))

        # make sure the gradient is finite to stop the minimizer getting upset
        if not all(np.isfinite(self.dnlZ)):
            bad = np.where(np.logical_not(np.isfinite(self.dnlZ)))
            for b in bad:
                self.dnlZ[b] = np.sign(self.dnlZ[b]) / np.finfo(float).eps

        if self.verbose:
            print("dnlZ= ", self.dnlZ, " | hyp=", hyp)

        return self.dnlZ

    # model estimation (optimization)
    def estimate(self, hyp0, covfunc, X, y, optimizer='cg'):
        """ Function to estimate the model
        """

        if optimizer.lower() == 'cg':  # conjugate gradients
            out = optimize.fmin_cg(self.loglik, hyp0, self.dloglik,
                                   (covfunc, X, y), disp=True, gtol=self.tol,
                                   maxiter=self.n_iter, full_output=1)

        elif optimizer.lower() == 'powell':  # Powell's method
            out = optimize.fmin_powell(self.loglik, hyp0, (covfunc, X, y),
                                       full_output=1)
        else:
            raise ValueError("unknown optimizer")

        self.hyp = out[0]
        self.nlZ = out[1]
        self.optimizer = optimizer

        return self.hyp

    def predict(self, hyp, X, y, Xs):
        """ Function to make predictions from the model
        """

        if self._updatepost(hyp, self.covfunc):
            self.post(hyp, self.covfunc, X, y)

        # hyperparameters
        sn2 = np.exp(2*hyp[0])     # noise variance
        theta = hyp[1:]            # (generic) covariance hyperparameters

        Ks = self.covfunc.cov(theta, Xs, X)
        kss = self.covfunc.cov(theta, Xs)

        # predictive mean
        ymu = Ks.dot(self.alpha)

        # predictive variance (for a noisy test input)
        v = solve(self.L, Ks.T)
        ys2 = kss - v.T.dot(v) + sn2

        return ymu, ys2
