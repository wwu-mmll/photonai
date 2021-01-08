from photonai.modelwrapper.keras_dnn_classifier import KerasDnnClassifier
from photonai.modelwrapper.keras_dnn_regressor import KerasDnnRegressor
from test.modelwrapper_tests.test_base_model_wrapper import BaseModelWrapperTest


class KerasDnnClassifierTest(BaseModelWrapperTest):

    def setUp(self):
        self.model_wrapper = KerasDnnClassifier()
        self.dnn = self.model_wrapper

    def test_multi_class(self):

        self.dnn = KerasDnnClassifier(multi_class=True, loss="")
        self.assertEqual(self.dnn.loss, "categorical_crossentropy")
        self.assertEqual(self.dnn.multi_class, True)
        self.dnn.multi_class = False
        self.assertEqual(self.dnn.loss, "binary_crossentropy")
        self.assertEqual(self.dnn.multi_class, False)

        with self.assertRaises(ValueError):
            self.dnn = KerasDnnClassifier(multi_class=True, loss='hinge')

        with self.assertRaises(ValueError):
            self.dnn = KerasDnnClassifier(multi_class=False, loss='kullback_leibler_divergence')

    def test_parameter_length(self):

        self.dnn = KerasDnnClassifier(hidden_layer_sizes=[1,2,3,4,5,6], dropout_rate=0.2, activations='tanh')
        self.assertEqual(self.dnn.dropout_rate, [0.2]*6)
        self.assertEqual(self.dnn.activations, ['tanh']* 6)

        self.dnn.dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual(self.dnn.dropout_rate, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        self.dnn.activations = ["tanh", "sigmoid"]*3
        self.assertEqual(self.dnn.activations, ["tanh", "sigmoid"]*3)

        self.dnn.dropout_rate = 0.4
        self.assertEqual(self.dnn.dropout_rate, [0.4] * 6)

        with self.assertRaises(ValueError):
            self.dnn.activations = "roundabout"

        with self.assertRaises(ValueError):
            self.dnn.hidden_layer_sizes = [1,2,3]

        with self.assertRaises(ValueError):
            self.dnn.dropout_rate = [0.2, 0.6]


class KerasDnnRegressorTest(KerasDnnClassifierTest):

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

        with self.assertRaises(ValueError):
            self.dnn = KerasDnnRegressor()
            self.dnn.target_activation = "relu"
