import unittest
import numpy as np
import warnings

from photonai.optimization import FloatRange, IntegerRange, Categorical, BooleanSwitch
from photonai.optimization.hyperparameters import NumberRange


class HyperparameterBaseTest(unittest.TestCase):

    def setUp(self):
        """
        Set default start setting for all tests.
        """
        self.intger_range = IntegerRange(2,6)
        self.float_range = FloatRange(0.1, 5.7)
        self.cateogrical_truth = ["a","b","c","d","e","f","g","h"]
        self.categorical = Categorical(self.cateogrical_truth)
        self.bool = BooleanSwitch()

    def test_rand_success(self):

        for _ in range(100):
            self.assertIn(self.intger_range.get_random_value(), list(range(2,6)))

            self.assertGreaterEqual(self.float_range.get_random_value(), 0.1)
            self.assertLess(self.float_range.get_random_value(), 5.7)

            self.assertIn(self.categorical.get_random_value(), self.cateogrical_truth)

            self.assertIn(self.bool.get_random_value(), [True, False])

        self.float_range.transform()
        self.intger_range.transform()

        for _ in range(100):
            self.assertIn(self.intger_range.get_random_value(definite_list=True), self.intger_range.values)
            self.assertIn(self.float_range.get_random_value(definite_list=True), self.float_range.values)

    def test_domain(self):

        self.float_range.transform()
        self.intger_range.transform()
        self.assertListEqual(self.intger_range.values, list(np.arange(2,6)))
        self.assertListEqual(self.float_range.values, list(np.linspace(0.1, 5.7, dtype=np.float64)))

        big_float_range = FloatRange(-300.57, np.pi*4000)
        big_float_range.transform()
        self.assertListEqual(big_float_range.values, list(np.linspace(-300.57, np.pi*4000)))
        self.assertListEqual(self.categorical.values, ["a","b","c","d","e","f","g","h"])
        self.assertListEqual(self.bool.values, [True, False])

    def test_rand_error(self):
        with self.assertRaises(ValueError):
            self.intger_range.get_random_value(definite_list=True)
        with self.assertRaises(ValueError):
            self.float_range.get_random_value(definite_list=True)
        with self.assertRaises(NotImplementedError):
            self.categorical.get_random_value(definite_list=False)
        with self.assertRaises(NotImplementedError):
            self.categorical.get_random_value(definite_list=False)

    def test_categorical(self):
        self.assertEqual(self.categorical[2], self.cateogrical_truth[2])


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
        np.testing.assert_array_almost_equal(number_range_logspace.values,  np.logspace(-1, 1, num=50))

        # error tests
        with self.assertRaises(ValueError):
            number_range = IntegerRange(start=0, stop=self.end, range_type="geomspace")
            number_range.transform()

        number_range = FloatRange(start=0.1, stop=self.end, range_type="geomspace")
        number_range.transform()

        with self.assertRaises(ValueError):
            number_range = IntegerRange(start=1, stop=15, range_type="logspace")
            number_range.transform()
        number_range = FloatRange(start=0.1, stop=self.end, range_type="logspace")
        number_range.transform()

        with self.assertRaises(ValueError):
            IntegerRange(start=self.start, stop=self.end, range_type="ownspace")

    def test_false_range_type(self):
        with self.assertRaises(ValueError):
            float_range = FloatRange(1.0, 5.2, range_type='normal_distributed')
            float_range.transform()

class HyperparameterOtherTest(unittest.TestCase):

    def test_categorical(self):
        """
        Test for class Categorical.
        """
        items = "Lorem ipsum dolor sit amet consetetur sadipscing elitr".split(" ")
        categorical = Categorical(values=items)
        self.assertEqual(categorical[2], "dolor")
        self.assertListEqual(categorical[2:5], items[2:5])
        self.assertListEqual(categorical.values, items)

    def test_boolean_switch(self):
        """
        Test for class BooleanSwitch.
        """
        boolean_switch = BooleanSwitch()
        self.assertListEqual(boolean_switch.values, [True, False])

    def test_start_stop_problem(self):
        with warnings.catch_warnings(record=True) as w:
            integer_range = IntegerRange(2, 1)
            integer_range.transform()
            assert any("NumberRange or one of its subclasses is empty" in s for s in [e.message.args[0] for e in w])

    def test_dtypes(self):
        complex_range = NumberRange(1, 5, num_type=complex, range_type="range")
        complex_range.transform()
        str_range = NumberRange(1, 2, num_type=np.bool_, range_type="range")
        str_range.transform()
