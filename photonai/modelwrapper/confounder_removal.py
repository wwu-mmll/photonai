import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from photonai.photonlogger.logger import logger

from typing import Union


class ConfounderRemoval(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, standardize_covariates: bool = True, confounder_names: Union[str, list] = 'confounder'):

        self.needs_covariates = True
        self.needs_y = False
        self.standardize_covariates = standardize_covariates
        if isinstance(confounder_names, str):
            self.confounder_names = [confounder_names]
        else:
            self.confounder_names = confounder_names
        self.scalers = list()
        self.olsModel_params = None

    def _check_for_confounders(self, kwargs):
        if len(self.confounder_names) == 1:
            if self.confounder_names[0] in kwargs:
                sample_ols_confounder = kwargs[self.confounder_names[0]]
                return sample_ols_confounder
            else:
                raise KeyError("Variable 'confounder' not found in kwargs dictionary")
        elif len(self.confounder_names) > 1:
            sample_ols_confounder = list()
            len_list = list()
            for confound in self.confounder_names:
                if confound in kwargs:
                    sample_ols_confounder.append(kwargs[confound])
                    len_list.append(len(kwargs[confound]))
                else:
                    raise KeyError("Variable '{}' not found in kwargs dictionary".format(confound))
            if len(set(len_list)) != 1:
                raise ValueError("Provided confounders do not match in length")
            return np.stack(sample_ols_confounder, axis=1)

    def _validate_dimension(self, X, sample_ols_confounder):
        if X.shape[0] != sample_ols_confounder.shape[0]:
            err_msg = 'Number of samples (N=' + str(X.shape[0]) + ') is not the same as number of cases (N=' + str(
                sample_ols_confounder.shape[0]) + ') in confounder!'
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _standardize(self, covariates, is_fit):
        logger.debug('Standardizing confounder prior to removal.')
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

        sample_ols_confounder = self._check_for_confounders(kwargs)
        self._validate_dimension(X, sample_ols_confounder)

        # standardize covariates
        if self.standardize_covariates:
            sample_ols_confounder = self._standardize(sample_ols_confounder, is_fit=True)

        # sample_ols_confounder: confounder variables of the samples to be fitted
        ols_confounder = sm.add_constant(sample_ols_confounder, has_constant="add")
        self.olsModel_params = np.zeros(shape=(X.shape[1], ols_confounder.shape[1]))

        for i in range(X.shape[1]):
            ols_model = sm.OLS(endog=np.squeeze(X[:, i]), exog=ols_confounder).fit()
            self.olsModel_params[i] = ols_model.params
        return self

    def transform(self, X, y=None, **kwargs):
        logger.debug('Regress out confounder.')
        sample_ols_confounder = self._check_for_confounders(kwargs)
        self._validate_dimension(X, sample_ols_confounder)

        # standardize covariates
        if self.standardize_covariates:
            sample_ols_confounder = self._standardize(sample_ols_confounder, is_fit=False)

        sample_ols_confounder = sm.add_constant(sample_ols_confounder, has_constant="add")
        X_new = np.empty(X.shape)
        for i in range(X.shape[1]):
            preds = np.matmul(sample_ols_confounder, np.squeeze(self.olsModel_params[i]))
            residuum_feature_vector = np.squeeze(X[:, i]) - preds
            # residuum_feature_vector += self.olsModel_params[i, 0]  # add intercept
            X_new[:, i] = np.asarray(residuum_feature_vector)  # writing back the residuum of the feature vector
        return X_new, kwargs

