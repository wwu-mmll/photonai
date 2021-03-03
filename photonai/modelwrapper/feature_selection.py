import numpy as np
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, f_classif, SelectPercentile, VarianceThreshold

from photonai.photonlogger.logger import logger


class FRegressionFilterPValue(BaseEstimator, TransformerMixin):
    """Feature Selection for Regression - p-value based.

    Fit f_regression and select all columns
    when p_value of column < p_threshold.

    """
    _estimator_type = "transformer"

    def __init__(self, p_threshold: float = .05):
        """
        Initialize the object.

        Parameters:
            p_threshold:
                Upper bound for p_values.

        """
        self.p_threshold = p_threshold
        self.selected_indices = []
        self.n_original_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calculation of the important columns.

        Apply f_regression on input X, y to generate p_values.
        selected_indices = all p_value(columns) < p_threshold.

        Parameters:
            X:
                The input samples of shape [n_samples, n_original_features]

            y:
                The input targets of shape [n_samples, 1]

        """
        self.n_original_features = X.shape[1]
        _, p_values = f_regression(X, y)
        self.selected_indices = np.where(p_values < self.p_threshold)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduced input X to selected_columns.

        Parameters:
            X
                The input samples of shape [n_samples, n_original_features]

        Returns:
            Column-filtered array of shape [n_samples, n_selected_features].

        """
        return X[:, self.selected_indices]

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse to original dimension.

        Parameters:
            X:
                The input samples of shape [n_samples, n_selected_features].

        Raises:
            ValueError: If input X has a different shape than during fitting.

        Returns:
            Array of shape [n_samples, n_original_features]
            with columns of zeros inserted where features would have
            been removed.

        """
        if X.shape[1] != len(self.selected_indices):
            msg = "X has a different shape than during fitting."
            logger.error(msg)
            raise ValueError(msg)

        Xt = np.zeros((X.shape[0], self.n_original_features))
        Xt[:, self.selected_indices] = X
        return Xt


class FRegressionSelectPercentile(BaseEstimator, TransformerMixin):
    """Feature Selection for regression data - percentile based.

    Apply VarianceThreshold -> SelectPercentile to data.
    SelectPercentile based on f_regression and parameter percentile.

    """
    _estimator_type = "transformer"

    def __init__(self, percentile: float = 10):
        """
        Initialize the object.

        Parameters:
            percentile:
                Percent of features to keep.

        """
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

    def inverse_transform(self, X):
        Xt = self.my_fs.inverse_transform(X)
        return self.var_thres.inverse_transform(Xt)


class FClassifSelectPercentile(BaseEstimator, TransformerMixin):
    """Feature Selection for classification data - percentile based.

    Apply VarianceThreshold -> SelectPercentile to data.
    SelectPercentile based on f_classif and parameter percentile.

    """
    _estimator_type = "transformer"

    def __init__(self, percentile: float = 10):
        """
        Initialize the object.

        Parameters:
            percentile:
                Percent of features to keep.

        """
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=f_classif, percentile=self.percentile)
        self.my_fs.fit(X, y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse to original dimension.

        1. SelectPercentile.inverse_transform
        2. VarianceThreshold.inverse_transform

        Parameters:
            X:
                The input samples of shape [n_samples, n_selected_features].

        Returns:
            Array of shape [n_samples, n_original_features]
            with columns of zeros inserted where features would have
            been removed.

        """
        Xt = self.my_fs.inverse_transform(X)
        return self.var_thres.inverse_transform(Xt)


class ModelSelector(BaseEstimator, TransformerMixin):
    """Model Selector - based on feature_importance.

    Apply feature selection on specific estimator
    and its importance scores.

     """
    _estimator_type = "transformer"

    def __init__(self, estimator_obj: BaseEstimator, threshold: float = 1e-5, percentile: bool = False):
        """
        Initialize the object.

        Parameters:
            estimator_obj:
                Estimator with fit/tranform and possibility of feature_importance.

            threshold:
                If percentile == True:
                    Lower Bound for required importance score to keep.
                If percentile == True:
                    percentage to keep (ordered features by feature_importance)

            percentile:
                Percent of features to keep.

        """
        self.threshold = threshold
        self.estimator_obj = estimator_obj
        self.selected_indices = []
        self.percentile = percentile
        self.importance_scores = []
        self.n_original_features = None

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
        self.n_original_features = X.shape[1]
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

    def inverse_transform(self, X):
        if X.shape[1] != len(self.selected_indices):
            msg = "X has a different shape than during fitting."
            logger.error(msg)
            raise ValueError(msg)
        Xt = np.zeros((X.shape[0], self.n_original_features))
        Xt[:, self.selected_indices] = X
        return Xt

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params['threshold']
            params.pop('threshold')
        self.estimator_obj.set_params(**params)

    def get_params(self, deep=True):
        return self.estimator_obj.get_params(deep)


class LassoFeatureSelection(BaseEstimator, TransformerMixin):
    """Lasso based feature selection - based on feature_importance.

    Apply Lasso to ModelSelection.

    """
    def __init__(self, percentile: float = 0.3, alpha: float = 1., **kwargs):
        """
        Initialize the object.

        Parameters:
            percentile: bool, default=False
                Percent of features to keep.

            alpha: float, default=1.
                Weighting parameter for Lasso.

            **kwargs:
                Passed to Lasso object.

        """
        self.percentile = percentile
        self.alpha = alpha
        self.model_selector = None
        self.Lasso_kwargs = kwargs
        self.needs_covariates=False
        self.needs_y = False

    def fit(self, X, y=None, **kwargs):
        self.model_selector = ModelSelector(Lasso(alpha=self.alpha, **self.Lasso_kwargs),
                                            threshold=self.percentile, percentile=True)

        self.model_selector.fit(X, y, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        selected_features = self.model_selector.transform(X, y, **kwargs)
        return selected_features

    def set_params(self, **params):
        super(LassoFeatureSelection, self).set_params(**params)

    def inverse_transform(self, X, y=None, **kwargs):
        return self.model_selector.inverse_transform(X)
