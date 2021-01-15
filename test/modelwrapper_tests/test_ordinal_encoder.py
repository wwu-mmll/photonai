import unittest
import numpy as np

from numpy.testing import assert_array_equal
from photonai.modelwrapper.OrdinalEncoder import FeatureEncoder


class FeatureEncoderTests(unittest.TestCase):

    def setUp(self):
        self.ordinal_encoder = FeatureEncoder()
        self.X = np.array([["a", 3, "b"], ["x", 5, "b"], ["a", 3, "c"]], dtype=object)
        self.values = np.array([1, 2, 1])
        self.classes = np.unique(self.values)

        self.intX = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

        self.X_with_bool = np.array([[1, 2, True], [24, 5, False]], dtype=object)

    def test_fit(self):
        self.ordinal_encoder.fit(self.X, self.values)
        self.assertEqual(len(self.ordinal_encoder.encoder_list), 3)
        self.assertIsNone(self.ordinal_encoder.encoder_list[1])

    def test_transform(self):
        self.ordinal_encoder.fit(self.X, self.values)
        result = self.ordinal_encoder.transform(np.array([["a",1, "b"], ["a", 1, "b"], ["a", 1, "b"]],  dtype=object))
        assert_array_equal(result,  np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=object))

    def test_inverse_transform(self):
        self.ordinal_encoder.fit(self.X, self.values)
        result = self.ordinal_encoder.inverse_transform(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=object))
        assert_array_equal(result, np.array([["a", 1, "b"], ["a", 1, "b"], ["a", 1, "b"]], dtype=object))

    def test_fit_transform(self):
        result = self.ordinal_encoder.fit_transform(self.X, self.values)
        assert_array_equal(result, np.array([[0, 3, 0], [1, 5, 0], [0, 3, 1]], dtype=object))

        result = self.ordinal_encoder.fit_transform(self.intX)
        assert_array_equal(result, self.intX)

        result = self.ordinal_encoder.fit_transform(np.array([["a", 1, "b"], ["a", 1, "b"], ["a", 1, "b"]],
                                                             dtype=object))
        assert_array_equal(result, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=object))

    def only_strings(self):
        result = self.ordinal_encoder.fit_transform(self.X_with_bool)
        assert_array_equal(result, self.X_with_bool)




