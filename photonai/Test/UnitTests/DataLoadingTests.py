import unittest

import numpy as np

from DataLoading.DataContainer import DataContainer, Features


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        self.dc = DataContainer()

    def tearDown(self):
        pass

    def test_fileNotFound(self):
        with self.assertRaises(FileNotFoundError):
            self.dc += Features('nonExistentFile.mat')

    def test_formatNotSupported(self):
        with self.assertRaises(TypeError):
            self.dc += Features('nonExistentFormat.sjwueiw')

    def test_csvLoading(self):
        self.dc += Features('TestFiles/testfile.csv')
        feature_object = self.dc.features
        # test if first column values are loaded correctly
        first_column_values = feature_object.data['A'].values
        expected_values = np.array([1, 2, 3, 4, 5])
        self.assertTrue(np.array_equal(first_column_values, expected_values))

    # def test_parameter(selfs):


    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
