from photonai.base import ClassificationPipe, ClassifierSwitch, RegressionPipe, RegressorSwitch
from photonai.helper.photon_base_test import elements_to_dict, PhotonBaseTest
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
import os
from pathlib import Path


class ModelZooTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(ModelZooTests, cls).setUpClass()

    def test_kwargs_setup(self):
        name = 'super_duper_classifier'
        cp = ClassificationPipe(name=name,
                                inner_cv=ShuffleSplit(n_splits=3),
                                use_test_set=False,
                                dim_reduction=True)

        # Hyperpipe should have gotten inner_cv and use_test_set,
        # DefaultPipeline should have gotten dim_reduction
        self.assertEqual(cp.cross_validation.inner_cv.n_splits, ShuffleSplit(n_splits=3).n_splits)
        self.assertFalse(cp.cross_validation.use_test_set)
        self.assertEqual(cp.elements[1].name, 'PCA')

    def test_default_pipeline_setup(self):
        cp1 = ClassificationPipe(add_default_pipeline_elements=True,
                                 scaling=True,
                                 dim_reduction=True,
                                 n_pca_components=10,
                                 feature_selection=True,
                                 imputation=True,
                                 imputation_nan_value=-99,
                                 add_estimator=True,
                                 default_estimator='SVC')
        self.assertEqual(cp1.elements[0].name, 'StandardScaler')
        self.assertEqual(cp1.elements[1].name, 'SimpleImputer')
        self.assertEqual(cp1.elements[2].name, 'FClassifSelectPercentile')
        self.assertEqual(cp1.elements[3].name, 'PCA')
        self.assertEqual(cp1.elements[3].base_element.n_components, 10)
        self.assertEqual(cp1.elements[4].name, 'SVC')

        cp2 = ClassificationPipe(add_default_pipeline_elements=False,
                                 scaling=True,
                                 dim_reduction=True,
                                 n_pca_components=10,
                                 feature_selection=True,
                                 imputation=True,
                                 imputation_nan_value=-99,
                                 add_estimator=True,
                                 default_estimator='SVC')

        self.assertEqual(len(cp2.elements), 0)

        with self.assertRaises(ValueError):
            cp_svc = ClassificationPipe(default_estimator=SVC)

    def test_data_loading(self):
        test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('test')[0], 'test')
        data_folder = Path(test_directory, 'base_tests/custom_elements/')
        rp_path = RegressionPipe(name='my_custom_regression_pipe',
                                 X_csv_path=data_folder.joinpath('breast_cancer_X.csv'),
                                 y_csv_path=data_folder.joinpath('breast_cancer_y.csv'))
        rp_path.fit()

        X, y = load_breast_cancer(return_X_y=True)
        with self.assertRaises(ValueError):
            rp_path.fit(X, y)

        with self.assertRaises(ValueError):
            rp_none = RegressionPipe()
            rp_none.fit()

        rp_numpy = RegressionPipe(name='custom_pipe2')
        rp_numpy.fit(X, y)

    def test_classifier_defaults(self):
        cl = ClassificationPipe()
        for m in ['balanced_accuracy', 'precision', 'recall']:
            self.assertTrue(m in cl.optimization.metrics)
        self.assertEqual(cl.optimization.best_config_metric, 'balanced_accuracy')
        self.assertIsInstance(cl.elements[-1], ClassifierSwitch)

    def test_regressor_defaults(self):
        reg = RegressionPipe()
        for r in ['mean_squared_error', 'mean_absolute_error', 'explained_variance']:
            self.assertTrue(r in reg.optimization.metrics)
        self.assertEqual(reg.optimization.best_config_metric, 'mean_squared_error')
        self.assertIsInstance(reg.elements[-1], RegressorSwitch)

