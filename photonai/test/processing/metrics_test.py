import unittest
import types
import numpy as np

from photonai.processing.metrics import Scorer


class ScorerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for Scorer Tests.
        """
        self.all_implemented_metrics = Scorer.ELEMENT_DICTIONARY.keys()
        self.some_not_implemented_metrics = ["abc_metric", "photon_metric"]

    def test_create(self):
        """
        Test for method create.
        Should searching for the metric by name and instantiates the according calculation function
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIsInstance(Scorer.create(implemented_metric), types.FunctionType)

        for not_implemented_metric in self.some_not_implemented_metrics:
            self.assertIsNone(Scorer.create(not_implemented_metric))

    def test_greater_is_better_distinction(self):
        """
        Test for method greater_is_better_distinction.
        Should return Boolean or raise NotImplementedError.
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIn(Scorer.greater_is_better_distinction(implemented_metric), [True, False])

        for not_implemented_metric in self.some_not_implemented_metrics:
            with self.assertRaises(NameError):
                Scorer.greater_is_better_distinction(not_implemented_metric)

    def test_calculate_metrics(self):
        """
        Test for method calculate_metrics.
        Handle all given metrics with a scorer call.
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIsInstance(Scorer.calculate_metrics([1, 1, 0, 1],
                                                           [0, 1, 0, 1],
                                                           [implemented_metric])[implemented_metric], float)

        for not_implemented_metric in self.some_not_implemented_metrics:
            np.testing.assert_equal(Scorer.calculate_metrics([1, 1, 0, 1],
                                                             [0, 1, 0, 1],
                                                             [not_implemented_metric])[not_implemented_metric], np.nan)
    
