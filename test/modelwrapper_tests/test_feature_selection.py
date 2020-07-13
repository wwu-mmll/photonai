from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base import Hyperpipe, PipelineElement, OutputSettings
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
        settings = OutputSettings(project_folder=self.tmp_folder_path)
        self.pipe_classif = Hyperpipe("feature_selection_pipe_classif",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, random_state=15),
                              metrics=["accuracy"], best_config_metric="accuracy",
                              output_settings=settings)
        self.pipe_regr = Hyperpipe("feature_selection_pipe_regr",
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=15),
                              inner_cv= KFold(n_splits=3, random_state=15),
                              metrics=["mean_absolute_error"], best_config_metric="mean_absolute_error",
                              output_settings=settings)

    def test_FRegressionFilterPValue(self):
        self.pipe_regr += PipelineElement('FRegressionFilterPValue')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_FRegressionSelectPercentile(self):
        self.pipe_regr += PipelineElement('FRegressionSelectPercentile')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)

    def test_FClassifSelectPercentile(self):
        self.pipe_classif += PipelineElement('FClassifSelectPercentile')
        self.pipe_classif += PipelineElement('SVC')
        self.pipe_classif.fit(self.X_classif, self.y_classif)

    def test_LassoFeatureSelection(self):
        self.pipe_regr += PipelineElement('LassoFeatureSelection')
        self.pipe_regr += PipelineElement('SVR')
        self.pipe_regr.fit(self.X_regr, self.y_regr)