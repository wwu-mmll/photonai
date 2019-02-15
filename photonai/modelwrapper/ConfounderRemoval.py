import os
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from photonai.photonlogger.Logger import Logger
from hashlib import sha1
from pathlib import Path


class ConfounderRemoval(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, standardize_covariates: bool = True, cache_dir=''):

        # check cache_dir
        if cache_dir and not os.path.isdir(cache_dir):
            try:
                os.mkdir(cache_dir)
                Logger().debug('Creating directory {}'.format(cache_dir))
            except:
                raise NotADirectoryError("{} is neither a directory nor could it be created.".format(cache_dir))


        self.needs_covariates = True
        self.needs_y = False
        self.cache_dir = cache_dir
        self.standardize_covariates = standardize_covariates
        self.scalers = list()
        self.olsModel_params = None

    def __check_for_covariates(self, kwargs):
        if "covariates" in kwargs:
            sample_ols_confounder = kwargs["covariates"]
            return sample_ols_confounder
        else:
            raise KeyError("'covariates' not found in kwargs dictionary")

    def __validate_dimension(self, X, sample_ols_confounder):

        if X.shape[0] != sample_ols_confounder.shape[0]:
            err_msg = 'Number of samples (N=' + str(X.shape[0]) + ') is not the same as number of cases (N=' + str(
                sample_ols_confounder.shape[0]) + ') in covariates!'
            Logger().error(err_msg)
            raise ValueError(err_msg)

    def _standardize(self, covariates, is_fit):
        Logger().debug('Standardizing covariates before confounder removal.')
        scaled_covs = list()
        if is_fit:
            # standardize covariates
            for cov in covariates.T:
                self.scalers.append(StandardScaler())
                scaled_covs.append(self.scalers[-1].fit_transform(cov.reshape(-1, 1)).squeeze())
            scaled_covs = np.asarray(scaled_covs).T
        else:
            for i, cov in enumerate(covariates.T):
                scaled_covs.append(self.scalers[i].transform(cov.reshape(-1, 1)).squeeze())
            scaled_covs = np.asarray(scaled_covs).T
        return scaled_covs

    def fit(self, X, y=None, **kwargs):

        sample_ols_confounder = self.__check_for_covariates(kwargs)
        self.__validate_dimension(X, sample_ols_confounder)

        # prepare hashing and caching
        hash_data = sha1(np.asarray(X, order='C')).hexdigest()
        hash_covs = sha1(np.asarray(sample_ols_confounder, order='C')).hexdigest()


        hash_file = Path(str(self.cache_dir + '/' + hash_data + '_' + hash_covs + '_' + str(self.standardize_covariates) + '.npz'))

        if hash_file.is_file() and self.cache_dir:
            Logger().debug('Skip fitting GLMs for confounder removal. Using cache...')
        else:
            # standardize covariates
            if self.standardize_covariates:
                sample_ols_confounder = self._standardize(sample_ols_confounder, is_fit=True)

            # sample_ols_confounder: confounder variables of the samples to be fitted
            ols_confounder = sm.add_constant(sample_ols_confounder)
            self.olsModel_params = np.zeros(shape=(X.shape[1], ols_confounder.shape[1]))

            for i in range(X.shape[1]):
                ols_model = sm.OLS(endog=np.squeeze(X[:, i]), exog=ols_confounder).fit()
                self.olsModel_params[i] = ols_model.params
        return self

    def transform(self, X, y=None, **kwargs):
        Logger().debug('Regress out confounder.')
        sample_ols_confounder = self.__check_for_covariates(kwargs)
        self.__validate_dimension(X, sample_ols_confounder)

        # prepare hashing and caching
        hash_data = sha1(np.asarray(X, order='C')).hexdigest()
        hash_covs = sha1(np.asarray(sample_ols_confounder, order='C')).hexdigest()

        hash_file = Path(str(self.cache_dir + '/' + hash_data + '_' + hash_covs  + '_' + str(self.standardize_covariates) + '.npz'))

        if hash_file.is_file() and self.cache_dir:
            X_new = np.load(hash_file)['arr_0']
        else:
            # standardize covariates
            if self.standardize_covariates:
                sample_ols_confounder = self._standardize(sample_ols_confounder, is_fit=False)

            sample_ols_confounder = sm.add_constant(sample_ols_confounder)
            X_new = np.empty(X.shape)
            for i in range(X.shape[1]):
                preds = np.matmul(sample_ols_confounder, np.squeeze(self.olsModel_params[i]))
                residuum_feature_vector = np.squeeze(X[:, i]) - preds
                # residuum_feature_vector += self.olsModel_params[i, 0]  # add intercept
                X_new[:, i] = np.asarray(residuum_feature_vector)  # writing back the residuum of the feature vector
            if self.cache_dir:
                np.savez(hash_file, X_new)
        return X_new, kwargs

