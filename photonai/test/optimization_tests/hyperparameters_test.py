import unittest
import numpy as np

from photonai.optimization import FloatRange, IntegerRange, Categorical, BooleanSwitch


class NumberRangeTest(unittest.TestCase):

    def setUp(self):
        """
        Set default start setting for all tests.
        """
        self.start = np.random.randint(0, 5)
        self.end = np.random.randint(100, 105)

    def test_NumberRange(self):
        """
        Test for class IntegerRange and FloatRange.
        """
        dtypes = {int: np.int32, float: np.float32}

        expected_linspace = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        number_range_linspace = IntegerRange(start=1, stop=9, num=9, range_type="linspace")
        number_range_linspace.transform()
        self.assertListEqual(expected_linspace, number_range_linspace.values)

        expected_geomspace = [1, 10, 100, 1000, 10000]
        number_range_geomspace = IntegerRange(1, 10000, num=5, range_type="geomspace")
        number_range_geomspace.transform()
        self.assertListEqual(expected_geomspace, number_range_geomspace.values)

        number_range_range = IntegerRange(self.start, self.end, step=2, range_type="range")
        number_range_range.transform()
        self.assertListEqual(number_range_range.values, list(np.arange(self.start, self.end, 2)))

        number_range_logspace = FloatRange(-1, 1, num=50, range_type='logspace')
        number_range_logspace.transform()
        np.testing.assert_array_almost_equal(number_range_logspace.values,  np.logspace(-1, 1, num=50).tolist())

        # error tests
        with self.assertRaises(ValueError):
            number_range = IntegerRange(start=0, stop=self.end, range_type="geomspace")
            number_range.transform()

        with self.assertRaises(ValueError):
            number_range = IntegerRange(start=1, stop=15, range_type="logspace")
            number_range.transform()

        with self.assertRaises(ValueError):
            IntegerRange(start=self.start, stop=self.end, range_type="ownspace")


class HyperparameterOtherTest(unittest.TestCase):

    def test_categorical(self):
        """
        Test for class Categorical.
        """
        items = "Lorem ipsum dolor sit amet consetetur sadipscing elitr".split(" ")
        categorical = Categorical(values=items)
        self.assertListEqual(categorical.values, items)

    def test_boolean_switch(self):
        """
        Test for class BooleanSwitch.
        """
        boolean_switch = BooleanSwitch()
        self.assertListEqual(boolean_switch.values, [True, False])
