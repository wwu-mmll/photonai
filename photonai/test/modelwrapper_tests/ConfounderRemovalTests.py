import unittest
from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
import os


class ConfounderRemovalTests(unittest.TestCase):

    def setUp(self):

        self.X, self.y = load_breast_cancer(True)
        self.X_train = self.X[:100]
        self.y_train = self.y[:100]
        self.shuffle_split = ShuffleSplit(test_size=0.2, n_splits=1, random_state=15)
        self.pipe = Hyperpipe("confounder_pipe", outer_cv=self.shuffle_split, inner_cv= KFold(n_splits=3, random_state=15),
                              metrics=["accuracy"], best_config_metric="accuracy")
        self.pipe += PipelineElement("StandardScaler")
        self.cr = PipelineElement("ConfounderRemoval")
        self.pipe += self.cr
        self.pipe += PipelineElement("SVC")
        self.random_confounders = np.random.randn(self.X.shape[0], 1)

        # do confounder removal by hand
        self.multiple_confounders = np.random.randn(self.X.shape[0], 2) * 10
        ols_confounder = sm.add_constant(self.multiple_confounders)
        self.X_transformed = np.empty(self.X.shape)
        for i in range(self.X.shape[1]):
            # fit
            model = sm.OLS(endog=np.squeeze(self.X[:, i]), exog=ols_confounder).fit()
            # transform
            self.X_transformed[:, i] = np.asarray(np.squeeze(self.X[:, i]) - np.matmul(ols_confounder, np.squeeze(model.params)))

        # prepare caching
        self.X_train_transformed = np.empty(self.X_train.shape)
        self.confounder_train = self.multiple_confounders[:100]
        ols_confounder_train = sm.add_constant(self.confounder_train)
        for i in range(self.X_train.shape[1]):
            # fit
            model = sm.OLS(endog=np.squeeze(self.X_train[:, i]), exog=ols_confounder_train).fit()
            # transform
            self.X_train_transformed[:, i] = np.asarray(
                np.squeeze(self.X_train[:, i]) - np.matmul(ols_confounder_train, np.squeeze(model.params)))

        # prepare confounder removal with standardization of covariates
        scaled_covs = list()
        # standardize covariates
        for cov in self.multiple_confounders.T:
            scaler = StandardScaler()
            scaled_covs.append(scaler.fit_transform(cov.reshape(-1, 1)).squeeze())
        scaled_covs = np.asarray(scaled_covs).T
        scaled_covs = sm.add_constant(scaled_covs)
        self.X_transformed_standardized = np.empty(self.X.shape)
        for i in range(self.X.shape[1]):
            # fit
            model = sm.OLS(endog=np.squeeze(self.X[:, i]), exog=scaled_covs).fit()
            # transform
            self.X_transformed_standardized[:, i] = np.asarray(
                np.squeeze(self.X[:, i]) - np.matmul(scaled_covs, np.squeeze(model.params)))

    def tearDown(self):
        pass

    def test_multiple_confounders(self):
        self.cr.fit(self.X, self.y, **{'covariates': self.multiple_confounders})
        X_transformed = self.cr.transform(self.X, **{'covariates': self.multiple_confounders})
        np.testing.assert_array_almost_equal(X_transformed[0], self.X_transformed)

    def test_multiple_confounders_with_caching(self):
        cache_dir = os.path.dirname(os.path.realpath(__file__))
        cr = PipelineElement("ConfounderRemoval", {}, cache_dir=cache_dir, standardize_covariates=False)
        cr.fit(self.X, self.y, **{'covariates': self.multiple_confounders})

        # use transform to write data to cache
        cr.transform(self.X, **{'covariates': self.multiple_confounders})

        # now the cached data should be loaded
        X_transformed = cr.transform(self.X, **{'covariates': self.multiple_confounders})
        np.testing.assert_array_almost_equal(X_transformed[0], self.X_transformed)

        # fit again to imitate cross validation
        cr.fit(self.X_train, self.y_train, **{'covariates': self.confounder_train})
        X_train_transformed = cr.transform(self.X_train, **{'covariates': self.confounder_train})
        np.testing.assert_array_almost_equal(X_train_transformed[0], self.X_train_transformed)

        [os.remove(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if f.endswith(".npz")]

    def test_standardize_covariates(self):
        self.cr.fit(self.X, self.y, **{'covariates': self.multiple_confounders})
        X_transformed = self.cr.transform(self.X, **{'covariates': self.multiple_confounders})
        np.testing.assert_array_almost_equal(X_transformed[0], self.X_transformed_standardized)

    def test_use(self):
        self.pipe.fit(self.X, self.y, **{'covariates': self.random_confounders})
        trans_data = self.pipe.transform(self.X, **{'covariates': self.random_confounders})

    def test_dimensions(self):
        with self.assertRaises(ValueError):
            self.cr.fit(self.X, self.y, covariates=np.random.randn(self.X.shape[0]-10, 2))

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.cr.fit(self.X, self.y, covariate=np.random.randn(self.X.shape[0]-10, 2))

    def test_cache_dir(self):
        with self.assertRaises(NotADirectoryError):
            self.cr = PipelineElement("ConfounderRemoval", {}, cache_dir='/not_a_directory')