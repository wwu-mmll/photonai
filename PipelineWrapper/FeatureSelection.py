# Wrapper for Feature Selection (Select Percentile)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression, f_classif, SelectPercentile, \
    VarianceThreshold, mutual_info_classif, mutual_info_regression, SelectKBest, chi2
from scipy.stats import pearsonr, f_oneway

class PearsonFeatureSelector(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, p_threshold=.05):
        self.p_threshold = p_threshold
        self.selected_indices = []

    def fit(self, X, y):

        # corr_coef = []
        corr_p = []

        for i in range(X.shape[1]):
            feature = X[:, i]
            corr = pearsonr(feature, y)
            # corr_coef.append(corr[0])
            corr_p.append(corr[1])

        # corr_coef = np.array(corr_coef)
        corr_p = np.array(corr_p)
        self.selected_indices = np.where((corr_p <= self.p_threshold))[0]
        return self

    def transform(self, X):
        selected_features = X[:, self.selected_indices]
        return selected_features


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


class AnovaSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, percentile=10):
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def loc_anova(self, data, targets):
        fs = []
        ps = []
        for snpInd in range(data.shape[1]):
            a = targets[data[:, snpInd] == 1]
            b = targets[data[:, snpInd] == 2]
            c = targets[data[:, snpInd] == 3]
            f, p = f_oneway(a, b, c)
            fs.append(f)
            ps.append(p)

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=self.loc_anova, percentile=self.percentile)
        self.my_fs.fit(X, y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)


class AnovaSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, percentile=10):
        # self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def loc_anova(self, data, targets):
        fs = []
        ps = []
        for feature_ind in range(data.shape[1]):
            group_data = []
            uni_targets = np.unique(data[:, feature_ind])
            for feature_cat in uni_targets:
                group_data.append(targets[data[:, feature_ind] == feature_cat])
            f, p = f_oneway(*group_data)
            fs.append(f)
            ps.append(p)
        return fs, ps

    def inverse_transform(self, X):
        return self.my_fs.inverse_transform(X)

    def fit(self, X, y):
        # X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=self.loc_anova, percentile=self.percentile)
        self.my_fs.fit(X, y)
        return self

    def transform(self, X):
        # X = self.var_thres.transform(X)
        return_values = self.my_fs.transform(X)
        if return_values.size == 0:
            return_values = np.zeros(X.shape)
        return return_values


class MIClassifSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = 'transformer'

    def __init__(self, percentile=10):
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=mutual_info_classif, percentile=self.percentile)
        self.my_fs.fit(X, y)
        return self

    def transform(self, X):
        X = self.var_thres.transform(X)
        return self.my_fs.transform(X)


class MIRegressionSelectPercentile(BaseEstimator, TransformerMixin):
    _estimator_type = 'transformer'

    def __init__(self, percentile=10):
        self.var_thres = VarianceThreshold()
        self.percentile = percentile
        self.my_fs = None

    def fit(self, X, y):
        X = self.var_thres.fit_transform(X)
        self.my_fs = SelectPercentile(score_func=mutual_info_regression, percentile=self.percentile)
        self.my_fs.fit(X, y)
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