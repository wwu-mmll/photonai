import numpy as np

from photonai.modelwrapper.imbalanced_data_transformer import ImbalancedDataTransformer
from test.modelwrapper_tests.test_base_model_wrapper import BaseModelWrapperTest

from imblearn.over_sampling.tests import test_smote
from imblearn.combine.tests import test_smote_tomek
from imblearn.under_sampling._prototype_selection.tests import test_instance_hardness_threshold


class ImbalancedDataTransformTest(BaseModelWrapperTest):
    """
    Tests based on implemented version by imblearn.
    For original implementation see:
    https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/tests/
    """

    def setUp(self):
        super(ImbalancedDataTransformTest, self).setUp()
        self.model_wrapper = ImbalancedDataTransformer()

    def test_strategy(self):
        with self.assertRaises(ValueError):
            ImbalancedDataTransformer(method_name="something")


    def test_strategy_oversampling(self):
        """
        sample test of different functions based on imblearn implementation for oversampling methods.
        """
        sampling_strategy = {0: 9, 1: 12}
        imbalanced_data_transformer = ImbalancedDataTransformer(method_name='SMOTE',
                                                                sampling_strategy = {0: 9, 1: 12},
                                                                random_state = test_smote.RND_SEED)

        # test_sample_regular_half() -> smote
        X_resampled, y_resampled = imbalanced_data_transformer.fit_transform(test_smote.X, test_smote.Y)
        X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
            1.25192108, -0.22367336
        ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
                             -0.28162401, -2.10400981
                         ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
                             0.70472253, -0.73309052
                         ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
                             0.88407872, 0.35454207
                         ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
                             -0.18410027, -0.45194484
                         ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049],
                         [-0.41635887, -0.38299653], [0.08711622, 0.93259929],
                         [1.70580611, -0.11219234], [0.36784496, -0.1953161]])
        y_gt = np.array(
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
        test_smote.assert_allclose(X_resampled, X_gt, rtol=test_smote.R_TOL)
        test_smote.assert_array_equal(y_resampled, y_gt)

    def test_strategy_undersampling(self):
        """
        sample test of different functions based on imblearn implementation for undersampling methods.
        """
        imbalanced_data_transformer = ImbalancedDataTransformer(method_name='InstanceHardnessThreshold',
                                                                estimator=test_instance_hardness_threshold.ESTIMATOR,
                                                                sampling_strategy={0: 6, 1: 8},
                                                                random_state=test_instance_hardness_threshold.RND_SEED)

        X_resampled, y_resampled = imbalanced_data_transformer.fit_resample(test_instance_hardness_threshold.X,
                                                                            test_instance_hardness_threshold.Y)
        assert X_resampled.shape == (15, 2)
        assert y_resampled.shape == (15,)

    def test_strategy_combine(self):
        """
        sample test of different functions based on imblearn implementation for oversampling methods.
        """
        imbalanced_data_transformer = ImbalancedDataTransformer(method_name='SMOTETomek',
                                                              random_state=test_smote_tomek.RND_SEED)
        X_resampled, y_resampled = imbalanced_data_transformer.fit_resample(test_smote_tomek.X, test_smote_tomek.Y)
        X_gt = np.array(
            [
                [0.68481731, 0.51935141],
                [1.34192108, -0.13367336],
                [0.62366841, -0.21312976],
                [1.61091956, -0.40283504],
                [-0.37162401, -2.19400981],
                [0.74680821, 1.63827342],
                [0.61472253, -0.82309052],
                [0.19893132, -0.47761769],
                [1.40301027, -0.83648734],
                [-1.20515198, -1.02689695],
                [-0.23374509, 0.18370049],
                [-0.00288378, 0.84259929],
                [1.79580611, -0.02219234],
                [0.38307743, -0.05670439],
                [0.70319159, -0.02571667],
                [0.75052536, -0.19246518],
            ]
        )
        y_gt = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
        test_smote_tomek.assert_allclose(X_resampled, X_gt, rtol=test_smote_tomek.R_TOL)
        test_smote_tomek.assert_array_equal(y_resampled, y_gt)
