import os
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from photonai.photonlogger.Logger import Logger


class ConfounderRemoval(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, logs=''):

        self.needs_covariates = True
        self.needs_y = False

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

        self.ols_confounder = None
        self.olsModel_params = None

    def __check_for_covariates(self, kwargs):
        if "covariates" in kwargs:
            sample_ols_confounder = kwargs["covariates"]
            return sample_ols_confounder
        else:
            raise KeyError("covariates not found in kwargs dictionary")

    def __validate_dimension(self, X, sample_ols_confounder):

        if X.shape[0] != sample_ols_confounder.shape[0]:
            err_msg = 'Number of samples (N=' + str(X.shape[0]) + ') is not the same as number of cases (N=' + str(
                sample_ols_confounder.shape[0]) + ') in covariates!'
            Logger().error(err_msg)
            raise ValueError(err_msg)

    def fit(self, X, y=None, **kwargs):

        sample_ols_confounder = self.__check_for_covariates(kwargs)
        self.__validate_dimension(X, sample_ols_confounder)

        # sample_ols_confounder: confounder variables of the samples to be fitted
        self.ols_confounder = sm.add_constant(sample_ols_confounder)
        self.olsModel_params = np.zeros(shape=(X.shape[1], self.ols_confounder.shape[1]))

        for i in range(X.shape[1]):
            ols_model = sm.OLS(endog=np.squeeze(X[:, i]), exog=self.ols_confounder).fit(X=X)
            self.olsModel_params[i] = ols_model.params
        return self

    def transform(self, X, y=None, **kwargs):

        sample_ols_confounder = self.__check_for_covariates(kwargs)
        self.__validate_dimension(X, sample_ols_confounder)

        sample_ols_confounder = sm.add_constant(sample_ols_confounder)
        X_new = np.empty(X.shape)
        for i in range(X.shape[1]):
            preds = np.matmul(sample_ols_confounder, np.squeeze(self.olsModel_params[i]))
            residuum_feature_vector = np.squeeze(X[:, i]) - preds
            # residuum_feature_vector += self.olsModel_params[i, 0]  # add intercept
            X_new[:, i] = np.asarray(residuum_feature_vector)  # writing back the residuum of the feature vector
        return X_new, kwargs


