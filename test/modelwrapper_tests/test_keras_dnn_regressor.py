from photonai.modelwrapper.keras_dnn_regressor import KerasDnnRegressor
from ..modelwrapper_tests.test_keras_dnn_classifier import KerasDnnClassifierTest


class KerasDnnClassifierTest(KerasDnnClassifierTest):

    def setUp(self):
        self.model_wrapper = KerasDnnRegressor()
        self.dnn = self.model_wrapper

    def test_multi_class(self):

        self.dnn = KerasDnnRegressor(loss="mean_squared_error")
        self.assertEqual(self.dnn.loss, "mean_squared_error")
        self.dnn.loss = "mean_squared_logarithmic_error"
        self.assertEqual(self.dnn.loss, "mean_squared_logarithmic_error")

        with self.assertRaises(ValueError):
            self.dnn = KerasDnnRegressor(loss='hinge')

        with self.assertRaises(ValueError):
            self.dnn = KerasDnnRegressor(loss='kullback_leibler_divergence')
