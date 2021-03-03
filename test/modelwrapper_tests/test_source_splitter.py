import unittest
import numpy as np

from numpy.testing import assert_array_equal
from photonai.modelwrapper.source_splitter import SourceSplitter


class FeatureEncoderTests(unittest.TestCase):

    def setUp(self):
        self.source_splitter = SourceSplitter(column_indices=[1,2])
        self.X = np.array([["a", 3, "b"], ["x", 5, "b"], ["a", 3, "c"]], dtype=object)
        self.values = np.array([1, 2, 1])
        self.X_two_columns = np.array([[1, 2], [2, 3], [3, 4]])

    def test_fit(self):
        NotImplementedError()

    def test_transform(self):
        result = self.source_splitter.transform(self.X)
        assert_array_equal(result, np.array([[3, "b"], [5, "b"], [3, "c"]], dtype=object))

        # selber abfangen?
        with self.assertRaises(IndexError):
            self.source_splitter.transform(self.X_two_columns)

    def test_inverse_transform(self):
        NotImplementedError()

    def test_fit_transform(self):
        result = self.source_splitter.transform(self.X)
        assert_array_equal(result, np.array([[3, "b"], [5, "b"], [3, "c"]], dtype=object))




