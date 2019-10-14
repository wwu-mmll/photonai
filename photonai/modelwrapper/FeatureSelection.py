# Wrapper for Feature Selection (Select Percentile)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, f_classif, SelectPercentile, \
    VarianceThreshold, mutual_info_classif, mutual_info_regression, SelectKBest, chi2
from scipy.stats import pearsonr, f_oneway
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import Lasso
from hashlib import sha1
from pathlib import Path
import statsmodels.api as sm
import multiprocessing
import os
from photonai.photonlogger.logger import logger


class FRegressionFilterPValue(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, p_threshold=.05):
        self.p_threshold = p_threshold
        self.selected_indices = []

    def fit(self, X, y):
        f_values, p_values = f_regression(X, y)
        self.selected_indices = np.where(p_values < self.p_threshold)
        return self

    def transform(self, X):
        return X[:, self.selected_indices]


class FRegressionSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, percentile=10):
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=f_regression, percentile=self.percentile)
        self.my_fs.fit(X,y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)


class FClassifSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, percentile=10):
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=f_classif, percentile=self.percentile)
        self.my_fs.fit(X,y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)


class Chi2KBest(BaseEstimator, TransformerMixin):
    _estimator_type = 'transformer'

    def __init__(self, k=10):
        self.var_thres = VarianceThreshold()
        self.k = k
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectKBest(score_func=chi2, k=self.k)
        self.my_fs.fit(X, y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)


class ModelSelector(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, estimator_obj, threshold=1e-5, percentile=False):
        self.threshold = threshold
        self.estimator_obj = estimator_obj
        self.selected_indices = []
        self.percentile = percentile
        self.importance_scores = []

    def _get_feature_importances(self, estimator, norm_order=1):
        """Retrieve or aggregate feature importances from estimator"""
        importances = getattr(estimator, "feature_importances_", None)

        if importances is None and hasattr(estimator, "coef_"):
            if estimator.coef_.ndim == 1:
                importances = np.abs(estimator.coef_)

            else:
                importances = np.linalg.norm(estimator.coef_, axis=0,
                                             ord=norm_order)

        elif importances is None:
            raise ValueError(
                "The underlying estimator %s has no `coef_` or "
                "`feature_importances_` attribute. Either pass a fitted estimator"
                " to SelectFromModel or call fit before calling transform."
                % estimator.__class__.__name__)

        return importances

    def fit(self, X, y=None, **kwargs):
        # 1. fit estimator
        self.estimator_obj.fit(X, y)
        # penalty = "l1"
        self.importance_scores = self._get_feature_importances(self.estimator_obj)

        if not self.percentile:
            self.selected_indices = np.where(self.importance_scores >= self.threshold)[0]
        else:
            # Todo: works only for binary classification, not for multiclass
            if self.threshold > 1:
                raise ValueError("Threshold should not be greater than 1")
            ordered_importances = np.sort(self.importance_scores)
            if isinstance(X, list):
                X = np.array(X)
            index = int(np.floor((1-self.threshold) * X.shape[1]))
            percentile_thres = ordered_importances[index]
            self.selected_indices = np.where(self.importance_scores >= percentile_thres)[0]
            # Todo: sortieren und Grenze definieren und dann np.where
            pass
        return self

    def transform(self, X, y=None, **kwargs):

        if isinstance(X, list):
            X = np.array(X)

        X_new = X[:, self.selected_indices]

        # if no features were selected raise error
        if X_new.shape[1] == 0:
            print("No Features were selected from model, using all features")
            return X
        return X_new

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params['threshold']
            params.pop('threshold')
        self.estimator_obj.set_params(**params)

    def get_params(self, deep=True):
        return self.estimator_obj.get_params(deep)


class LassoFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, percentile_to_keep=0.3, alpha=1, **kwargs):

        self.percentile_to_keep = percentile_to_keep
        self.alpha = alpha
        self.model_selector = None
        self.Lasso_kwargs = kwargs
        self.needs_covariates=False
        self.needs_y = False

    def fit(self, X, y=None, **kwargs):
        self.model_selector = ModelSelector(Lasso(alpha=self.alpha, **self.Lasso_kwargs),
                                            threshold=self.percentile_to_keep, percentile=True)

        self.model_selector.fit(X, y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        selected_features = self.model_selector.transform(X, y, **kwargs)
        return selected_features

    def set_params(self, **params):
        super(LassoFeatureSelection, self).set_params(**params)
