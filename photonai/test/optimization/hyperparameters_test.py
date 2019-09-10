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
        for numberType in [int, float]:
            if numberType == float:
                number_range = FloatRange(start=self.start, stop=self.end)
            else:
                number_range = IntegerRange(start=self.start, stop=self.end)
            # range_type: range
            number_range.range_type = "range"
            for step in [1, 2, 3]:
                number_range.step = step
                number_range.transform()
                self.assertListEqual(number_range.values, list(np.arange(self.start, self.end, step)))

            number_range.step = 1
            # range_type: linspace
            number_range.range_type = "linspace"
            number_range.transform()
            self.assertEqual(number_range.values[-1], self.end)
            self.assertListEqual(number_range.values,
                                 list(set([numberType(x) for x in
                                           np.linspace(self.start, self.end, dtype=dtypes[numberType])])))

            # range_type: geomspace
            number_range.range_type = "geomspace"
            number_range.start += 1
            number_range.transform()
            self.assertListEqual(number_range.values,
                                 list(set([numberType(x) for x in
                                           np.geomspace(number_range.start, self.end, dtype=dtypes[numberType])])))

            # range_type: logspace
            number_range.range_type = "logspace"
            number_range.start = self.start / 1000
            number_range.stop = self.end / 1000
            number_range.transform()
            self.assertListEqual(number_range.values,
                                 list(set([numberType(x) for x in
                                           np.logspace(self.start / 1000, self.end / 1000, dtype=dtypes[numberType])])))

            # error tests
            with self.assertRaises(ValueError):
                number_range = IntegerRange(start=0, stop=self.end, range_type="geomspace")
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
