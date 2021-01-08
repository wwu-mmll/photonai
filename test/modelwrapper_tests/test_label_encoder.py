import unittest
import numpy as np

from numpy.testing import assert_array_equal
from photonai.modelwrapper.label_encoder import LabelEncoder



class LabelEncoderTests(unittest.TestCase):

    def setUp(self):
        self.label_encoder = LabelEncoder()
        self.X = np.random.rand(5,5)
        self.values = np.array(["a", "b", "a", "a", "a", "c"])
        self.classes = np.unique(self.values)

    def test_fit(self):
        self.label_encoder.fit(self.X, self.values)
        assert_array_equal(self.label_encoder.classes_, self.classes)

    def test_transform(self):
        self.label_encoder.fit(self.X, self.values)
        trans_X, trans_y = self.label_encoder.transform(self.X, self.values)
        assert_array_equal(trans_y, [0, 1, 0, 0, 0, 2])

        with self.assertRaises(ValueError):
            self.label_encoder.transform(self.X, "unknown")

    def test_inverse_transform(self):
        self.label_encoder.fit(self.X, self.values)
        trans_X, trans_y = self.label_encoder.transform(self.X, self.values)
        assert_array_equal(self.label_encoder.inverse_transform(trans_y), self.values)

    def test_fit_transform(self):
        ret = self.label_encoder.fit_transform(self.X, self.values)
        assert_array_equal(ret, [0, 1, 0, 0, 0, 2])










