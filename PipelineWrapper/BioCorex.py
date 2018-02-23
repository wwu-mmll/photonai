import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..Logging.Logger import Logger
import sys
sys.path.append()
from ..SidePackages.bio_corex import corex as bc


class BioCorex(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self, n_hidden=10, marginal_description='discrete', smooth_marginals=False, continuous_output=True):
        self.corex_model = None
        self.n_hidden = n_hidden
        self.marginal_description = marginal_description
        self.smooth_marginal = smooth_marginals
        self.continuous_output = continuous_output

    def fit(self, X, y):
        self.corex_model = bc.Corex(n_hidden=self.n_hidden, marginal_description=self.marginal_description,
                                    smooth_marginals=self.smooth_marginal)
        self.corex_model.fit(X)
        return self

    def transform(self, X):
        if self.continuous_output:
            X_transformed,_ = self.corex_model.transform(X, details=True)
        else:
            X_transformed = self.corex_model.transform(X, details=False)
        return X_transformed




