import numpy as np
import statsmodels.api as sm
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from photonai.base import Hyperpipe, PipelineElement
from photonai.helper.photon_base_test import PhotonBaseTest


class ConfounderRemovalTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(ConfounderRemovalTests, cls).setUpClass()

    def setUp(self):

        super(ConfounderRemovalTests, self).setUp()
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.X_train = self.X[:100]
        self.y_train = self.y[:100]
        self.shuffle_split = ShuffleSplit(test_size=0.2, n_splits=1, random_state=15)
        self.pipe = Hyperpipe("confounder_pipe",
                              outer_cv=self.shuffle_split, inner_cv= KFold(n_splits=3, shuffle=True, random_state=15),
                              metrics=["accuracy"], best_config_metric="accuracy", project_folder=self.tmp_folder_path)
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

        # prepare statistical testing of confounder removal
        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        x = norm.rvs(size=(4, 300))

        # desired covariance matrix
        r = np.array([
            [1, .9, .9, .9],
            [.9, 1, .9, .9],
            [.9, .9, 1, .9],
            [.9, .9, .9, 1],
        ])
        c = cholesky(r, lower=True)

        # convert the data to correlated random variables
        self.z = np.dot(c, x).T

    def test_confounder_removal_statistically(self):
        cr = PipelineElement("ConfounderRemoval", {}, standardize_covariates=False)
        cr.fit(self.z[:, 1:3], self.z[:, 0], **{'confounder': self.z[:, 3]})

        # use transform to write data to cache
        z_transformed = cr.transform(self.z[:, 1:3], **{'confounder': self.z[:,3]})
        corr = np.corrcoef(np.concatenate([self.z[:, 0].reshape(-1, 1), z_transformed[0],
                                           self.z[:, 3].reshape(-1, 1)], axis=1), rowvar=False)
        # correlation between target and feature should be lower than 0.25 in this case
        # correlation between covariate and feature should be near zero
        self.assertLess(corr[1, 0], 0.3)
        self.assertLess(corr[2, 0], 0.3)
        self.assertAlmostEqual(corr[3, 1], 0)
        self.assertAlmostEqual(corr[3, 2], 0)

    def test_multiple_confounders(self):
        self.cr.fit(self.X, self.y, **{'confounder': self.multiple_confounders})
        X_transformed = self.cr.transform(self.X, **{'confounder': self.multiple_confounders})
        np.testing.assert_array_almost_equal(X_transformed[0], self.X_transformed)

    def test_standardize_covariates(self):
        self.cr.fit(self.X, self.y, **{'confounder': self.multiple_confounders})
        X_transformed = self.cr.transform(self.X, **{'confounder': self.multiple_confounders})
        np.testing.assert_array_almost_equal(X_transformed[0], self.X_transformed_standardized)

    def test_use(self):
        self.pipe.fit(self.X, self.y, **{'confounder': self.random_confounders})
        trans_data = self.pipe.transform(self.X, **{'confounder': self.random_confounders})

    def test_dimensions(self):
        with self.assertRaises(ValueError):
            self.cr.fit(self.X, self.y, confounder=np.random.randn(self.X.shape[0]-10, 2))

    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.cr.fit(self.X, self.y, covariate=np.random.randn(self.X.shape[0]-10, 2))
