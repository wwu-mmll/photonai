import numpy as np

from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_breast_cancer, load_diabetes

from photonai import Hyperpipe, PipelineElement
from photonai.helper.photon_base_test import PhotonBaseTest

from photonai.modelwrapper.cpm_feature_selection import CPMFeatureSelection


class CPMFeatureSelectionTest(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(CPMFeatureSelectionTest, cls).setUpClass()

    def setUp(self):
        super(CPMFeatureSelectionTest, self).setUp()
        self.X_classif, self.y_classif = load_breast_cancer(return_X_y=True)
        self.X_regr, self.y_regr = load_diabetes(return_X_y=True)
        self.pipe_classif = Hyperpipe("cpm_feature_selection_pipe_classif",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, shuffle=True, random_state=15),
                              metrics=["accuracy"], best_config_metric="accuracy",
                              project_folder=self.tmp_folder_path)
        self.pipe_regr = Hyperpipe("cpm_feature_selection_pipe_regr",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, shuffle=True, random_state=15),
                              metrics=["mean_absolute_error"], best_config_metric="mean_absolute_error",
                              project_folder=self.tmp_folder_path)

    def test_cpm_regression(self):
        self.pipe_regr += PipelineElement('CPMFeatureSelection', hyperparameters={})
        self.pipe_regr += PipelineElement('LinearRegression')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_cpm_classification(self):
        self.pipe_classif += PipelineElement('CPMFeatureSelection',
                                             hyperparameters={'corr_method': ['pearson', 'spearman']})
        self.pipe_classif += PipelineElement('LogisticRegression')
        self.pipe_classif.fit(self.X_classif, self.y_classif)

    def test_columnwise_correlation(self):
        for cpm_corr_method, scipy_corr_method in [(CPMFeatureSelection._columnwise_pearson, pearsonr),
                                                   (CPMFeatureSelection._columnwise_spearman, spearmanr)]:
            r_values, p_values = cpm_corr_method(self.X_classif, self.y_classif)
            r_scipy_first = scipy_corr_method(self.X_classif[:, 0], self.y_classif)
            r_scipy_last = scipy_corr_method(self.X_classif[:, -1], self.y_classif)
            self.assertAlmostEqual(r_values[0], r_scipy_first.statistic)
            self.assertAlmostEqual(p_values[0], r_scipy_first.pvalue)
            self.assertAlmostEqual(r_values[-1], r_scipy_last.statistic)
            self.assertAlmostEqual(p_values[-1], r_scipy_last.pvalue)

    def test_cpm_inverse(self):
        cpm = PipelineElement('CPMFeatureSelection',
                              hyperparameters={'corr_method': ['pearson']})

        cpm.fit(self.X_classif, self.y_classif)
        X_transformed, _, _ = cpm.transform(self.X_classif)
        X_back, _, _ = cpm.inverse_transform(np.asarray([3, -3]))
        self.assertEqual(X_transformed.shape[1], 2)
        self.assertEqual(self.X_classif.shape[1], X_back.shape[1])
        self.assertEqual(np.min(X_back), -3)
        self.assertEqual(np.max(X_back), 3)

        with self.assertRaises(ValueError):
            cpm.inverse_transform(X_transformed)

        with self.assertRaises(ValueError):
            cpm.inverse_transform(X_transformed.T)

    def test_wrong_corr_method(self):
        with self.assertRaises(NotImplementedError):
            PipelineElement('CPMFeatureSelection', corr_method='Pearsons')

    def test_cpm_transform(self):
        element = PipelineElement('CPMFeatureSelection', hyperparameters={})
        element.fit(self.X_classif, self.y_classif)
        X_transformed, _, _ = element.transform(self.X_classif)
        self.assertEqual(X_transformed.shape[0], self.X_classif.shape[0])
        self.assertEqual(X_transformed.shape[1], 2)
