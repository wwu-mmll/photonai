from numpy.testing import assert_array_almost_equal
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement
from photonai.helper.photon_base_test import PhotonBaseTest


class FeatureSelectionTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(FeatureSelectionTests, cls).setUpClass()

    def setUp(self):
        super(FeatureSelectionTests, self).setUp()
        self.X_classif, self.y_classif = load_breast_cancer(return_X_y=True)
        self.X_regr, self.y_regr = load_boston(return_X_y=True)
        self.pipe_classif = Hyperpipe("feature_selection_pipe_classif",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, shuffle=True, random_state=15),
                              metrics=["accuracy"], best_config_metric="accuracy",
                              project_folder=self.tmp_folder_path)
        self.pipe_regr = Hyperpipe("feature_selection_pipe_regr",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, shuffle=True, random_state=15),
                              metrics=["mean_absolute_error"], best_config_metric="mean_absolute_error",
                              project_folder=self.tmp_folder_path)

    def test_FRegressionFilterPValue(self):
        self.pipe_regr += PipelineElement('FRegressionFilterPValue')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_FRegressionFilterPValue_inverse(self):
        frfpv = PipelineElement('FRegressionFilterPValue', p_threshold=0.001)
        frfpv.fit(self.X_regr[:30], self.y_regr[:30])
        X_selected, _, _ = frfpv.transform(self.X_regr)
        X_back, _, _ = frfpv.inverse_transform(X_selected)
        self.assertLess(X_selected.shape[1], X_back.shape[1])
        self.assertEqual(self.X_regr.shape, X_back.shape)
        assert_array_almost_equal(X_selected,  frfpv.transform(X_back)[0])

        with self.assertRaises(ValueError):
            frfpv.inverse_transform(X_selected[:, :int(X_selected.shape[1]*0.5)])

    def test_FRegressionSelectPercentile(self):
        self.pipe_regr += PipelineElement('FRegressionSelectPercentile')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_FRegressionSelectPercentile_inverse(self):
        frsp = PipelineElement('FRegressionSelectPercentile', percentile=5)
        frsp.fit(self.X_regr[:30], self.y_regr[:30])
        X_selected, _, _ = frsp.transform(self.X_regr)
        X_back, _, _ = frsp.inverse_transform(X_selected)
        self.assertLess(X_selected.shape[1], X_back.shape[1])
        self.assertEqual(self.X_regr.shape, X_back.shape)
        assert_array_almost_equal(X_selected,  frsp.transform(X_back)[0])

    def test_FClassifSelectPercentile(self):
        self.pipe_classif += PipelineElement('FClassifSelectPercentile')
        self.pipe_classif += PipelineElement('SVC')
        self.pipe_classif.fit(self.X_classif, self.y_classif)

    def test_FClassifSelectPercentile_inverse(self):
        fcsp = PipelineElement('FClassifSelectPercentile', percentile=5)
        fcsp.fit(self.X_classif[:30], self.y_classif[:30])
        X_selected, _, _ = fcsp.transform(self.X_classif)
        X_back, _, _ = fcsp.inverse_transform(X_selected)
        self.assertLess(X_selected.shape[1], X_back.shape[1])
        self.assertEqual(self.X_classif.shape, X_back.shape)
        assert_array_almost_equal(X_selected,  fcsp.transform(X_back)[0])

    def test_LassoFeatureSelection(self):
        self.pipe_regr += PipelineElement('LassoFeatureSelection')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_LassoFeatureSelection_inverse(self):
        lfs = PipelineElement('LassoFeatureSelection')
        lfs.fit(self.X_regr[:30], self.y_regr[:30])
        X_selected, _, _ = lfs.transform(self.X_regr)
        X_back, _, _ = lfs.inverse_transform(X_selected)
        self.assertLess(X_selected.shape[1], self.X_regr.shape[1])
        self.assertEqual(self.X_regr.shape, X_back.shape)
        assert_array_almost_equal(X_selected,  lfs.transform(X_back)[0])

        with self.assertRaises(ValueError):
            lfs.inverse_transform(X_selected[:, :int(X_selected.shape[1]*0.5)])
