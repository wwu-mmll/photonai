import unittest
import numpy as np

from photonai.optimization import DummyPerformance, MinimumPerformance
from photonai.processing.results_structure import MDBConfig, MDBScoreInformation, MDBInnerFold


class PerformanceConstraintsTest(unittest.TestCase):

    def setUp(self):
        """
        Set default start setting for all tests.
        """
        self.minimum_performance = MinimumPerformance(strategy='first', metric='f1_score', threshold=0)
        self.dummy_performance = DummyPerformance(strategy='first', metric='mean_squared_error', margin=0.1)

        metrics_list = ["f1_score", "mean_squared_error"]
        self.dummy_config_item = MDBConfig()
        self.dummy_config_item.inner_folds = []
        for i in range(5):
            inner_fold = MDBInnerFold()
            inner_fold.validation = MDBScoreInformation()
            for metric in metrics_list:
                inner_fold.validation.metrics[metric] = np.random.randint(0, 1)/2+0.0001
            self.dummy_config_item.inner_folds.append(inner_fold)

        self.dummy_linear_config_item = MDBConfig()
        self.dummy_linear_config_item.inner_folds = []
        for i in range(5):
            inner_fold = MDBInnerFold()
            inner_fold.validation = MDBScoreInformation()
            for metric in metrics_list:
                inner_fold.validation.metrics[metric] = i/4
            self.dummy_linear_config_item.inner_folds.append(inner_fold)

    def test_strategy(self):
        """
        Test for set different strategies.
        """
        # set after declaration
        with self.assertRaises(KeyError):
            self.minimum_performance.strategy = "standart"

        with self.assertRaises(KeyError):
            self.minimum_performance.strategy = 412

        self.assertEqual(self.minimum_performance.strategy.name, 'first')

        self.minimum_performance.strategy = 'mean'
        self.assertEqual(self.minimum_performance.strategy.name, 'mean')

        self.dummy_performance.strategy = 'all'
        self.assertEqual(self.dummy_performance.strategy.name, 'all')

        # set in declaration
        with self.assertRaises(KeyError):
            MinimumPerformance(strategy='overall', metric='f1_score')

        with self.assertRaises(KeyError):
            DummyPerformance(strategy='last', metric='f1_score')

    def test_greater_is_better(self):
        """
        Test for set different metrics (score/error).
        """
        # set after declaration
        self.assertEqual(self.minimum_performance._greater_is_better, True)

        self.minimum_performance.metric = "mean_squared_error"
        self.assertEqual(self.minimum_performance._greater_is_better, False)

        self.assertEqual(self.dummy_performance._greater_is_better, False)

        self.dummy_performance.metric = "f1_score"
        self.assertEqual(self.dummy_performance._greater_is_better, True)

    def test_shall_continue(self):
        """
        Test for shall_continue function.
        """
        # reutnrs every times False if the metric does not exists
        self.minimum_performance.metric = "own_metric"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_config_item), True)

        # dummy_item with random values
        # score
        self.minimum_performance.metric = "f1_score"
        self.minimum_performance.threshold = 0
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_config_item), True)

        self.minimum_performance.threshold = 1
        self.minimum_performance.strategy = "mean"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_config_item), False)

        # error
        self.minimum_performance.metric = "mean_squared_error"
        self.minimum_performance.threshold = 0
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_config_item), False)

        self.minimum_performance.threshold = 1
        self.minimum_performance.strategy = "mean"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_config_item), True)

        # dummy_item with linear values
        # score
        self.minimum_performance.metric = "f1_score"
        self.minimum_performance.threshold = 0.5
        self.minimum_performance.strategy = "first"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), False)
        self.minimum_performance.strategy = "mean"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), True)
        self.minimum_performance.strategy = "all"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), False)

        # error
        self.minimum_performance.metric = "mean_squared_error"
        self.minimum_performance.threshold = 0.5
        self.minimum_performance.strategy = "first"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), True)
        self.minimum_performance.strategy = "mean"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), True)
        self.minimum_performance.strategy = "all"
        self.assertEqual(self.minimum_performance.shall_continue(self.dummy_linear_config_item), False)

    def test_copy_me(self):
        """
        Test for copy_me function.
        """
        self.dummy_performance.set_dummy_performance(self.dummy_config_item.inner_folds[0])
        new_dummy_performance = self.dummy_performance.copy_me()
        self.assertDictEqual(new_dummy_performance.__dict__,self.dummy_performance.__dict__)

        new_minimum_performance = self.minimum_performance.copy_me()
        self.assertDictEqual(new_minimum_performance.__dict__,self.minimum_performance.__dict__)
