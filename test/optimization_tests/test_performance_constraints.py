import unittest
import numpy as np
import warnings

from photonai.optimization import DummyPerformanceConstraint, MinimumPerformanceConstraint, BestPerformanceConstraint, IntegerRange
from photonai.optimization.performance_constraints import PhotonBaseConstraint
from photonai.processing.results_structure import MDBConfig, MDBScoreInformation, MDBInnerFold

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from photonai.base import Hyperpipe, PipelineElement


class PhotonBaseConstraintTest(unittest.TestCase):

    def setUp(self):
        """Set default start setting for all tests."""
        self.constraint_object = PhotonBaseConstraint(strategy='first',
                                                      metric='mean_squared_error',
                                                      margin=0.1)

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
        """Test for set different strategies."""
        # set after declaration
        with self.assertRaises(KeyError):
            self.constraint_object.strategy = "normal"

        with self.assertRaises(KeyError):
            self.constraint_object.strategy = 412

        self.constraint_object.strategy = 'first'
        self.assertEqual(self.constraint_object.strategy.name, 'first')

        self.constraint_object.strategy = 'mean'
        self.assertEqual(self.constraint_object.strategy.name, 'mean')

        self.constraint_object.strategy = 'any'
        self.assertEqual(self.constraint_object.strategy.name, 'any')

        # set in declaration
        with self.assertRaises(KeyError):
            PhotonBaseConstraint(strategy='overall', metric='f1_score')

        with self.assertRaises(KeyError):
            PhotonBaseConstraint(strategy='last', metric='f1_score')

    def test_greater_is_better(self):
        """Test for set different metrics (score/error)."""
        # set after declaration
        self.constraint_object.metric = "mean_squared_error"
        self.assertEqual(self.constraint_object._greater_is_better, False)

        self.constraint_object.metric = "f1_score"
        self.assertEqual(self.constraint_object._greater_is_better, True)

    def test_shall_continue(self):
        """Test for shall_continue function."""
        # returns every times False if the metric does not exists
        with warnings.catch_warnings(record=True) as w:
            self.constraint_object.metric = "own_metric"
            self.assertTrue(self.constraint_object.shall_continue(self.dummy_config_item))
            assert any("The metric is not known." in s for s in [e.message.args[0] for e in w])

        with warnings.catch_warnings(record=True) as w:
            self.constraint_object.metric = "f1_score"
            self.constraint_object.threshold = None
            self.assertTrue(self.constraint_object.shall_continue(self.dummy_config_item))
            assert any("Could not established threshold for performance" in s for s in [e.message.args[0] for e in w])

        # dummy_item with random values
        # score
        self.constraint_object.metric = "f1_score"
        self.constraint_object.threshold = 0
        self.assertTrue(self.constraint_object.shall_continue(self.dummy_config_item))

        self.constraint_object.threshold = 1
        self.constraint_object.strategy = "mean"
        self.assertFalse(self.constraint_object.shall_continue(self.dummy_config_item))

    def test_copy_me(self):
        """Test for copy_me function."""
        new_constraint_object = self.constraint_object.copy_me()
        self.assertDictEqual(new_constraint_object.__dict__,self.constraint_object.__dict__)


class MinimumPerformanceTest(PhotonBaseConstraintTest):

    def setUp(self):
        super(MinimumPerformanceTest, self).setUp()
        self.constraint_object = MinimumPerformanceConstraint(strategy='first', metric='f1_score', threshold=0)

    def test_shall_continue(self):
        super(MinimumPerformanceTest, self).test_shall_continue()

        # error
        self.constraint_object.metric = "mean_squared_error"
        self.constraint_object.threshold = 0
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_config_item), False)

        self.constraint_object.threshold = 1
        self.constraint_object.strategy = "mean"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_config_item), True)

        # dummy_item with linear values
        # score
        self.constraint_object.metric = "f1_score"
        self.constraint_object.threshold = 0.5
        self.constraint_object.strategy = "first"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), False)
        self.constraint_object.strategy = "mean"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), True)
        self.constraint_object.strategy = "any"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), False)

        # error
        self.constraint_object.metric = "mean_squared_error"
        self.constraint_object.threshold = 0.5
        self.constraint_object.strategy = "first"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), True)
        self.constraint_object.strategy = "mean"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), True)
        self.constraint_object.strategy = "any"
        self.assertEqual(self.constraint_object.shall_continue(self.dummy_linear_config_item), False)

class DummyPerformanceConstraints(PhotonBaseConstraintTest):

    def setUp(self):
        super(DummyPerformanceConstraints, self).setUp()
        self.constraint_object = DummyPerformanceConstraint(strategy='first', metric='mean_squared_error', margin=0.1)
        self.constraint_object.set_dummy_performance(self.dummy_config_item.inner_folds[0])


class BestPerformanceTest(PhotonBaseConstraintTest):

    def setUp(self):
        super(BestPerformanceTest, self).setUp()
        self.constraint_object = BestPerformanceConstraint(strategy='mean', margin=-0.2, metric='mean_squared_error')

    def test_shall_continue(self):
        X, y = load_boston(return_X_y=True)

        inner_fold_length = 7
        my_pipe = Hyperpipe(name='performance_pipe',
                            optimizer='random_search',
                            optimizer_params={'limit_in_minutes': 0.2},
                            metrics=['mean_squared_error'],
                            best_config_metric='mean_squared_error',
                            inner_cv=KFold(n_splits=inner_fold_length),
                            use_test_set=True,
                            project_folder='./tmp',
                            verbosity=0,
                            performance_constraints=[self.constraint_object])

        my_pipe += PipelineElement('StandardScaler')
        my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})
        my_pipe.fit(X, y)

        # clip config results
        results = my_pipe.results.outer_folds[0].tested_config_list

        configs = []

        for i in range(len(results)-1):
            configs.append([x.validation.metrics['mean_squared_error'] for x in results[i].inner_folds])

        threshold = np.inf
        for val in configs[:10]:
            challenger = np.mean(val)
            if threshold > challenger:
                threshold = challenger

        originals_for_std = configs[:10]
        for i, val in enumerate(configs[10:]):
            std = np.mean([np.std(x) for x in originals_for_std])
            for j, v in enumerate(val):

                if np.mean(val[:j+1]) > threshold + std:
                    self.assertEqual(v, val[-1])
                    continue
                if len(val) == inner_fold_length-1 and np.mean(val)<threshold+std:
                    threshold = np.mean(val)
            if len(val)>1:
                originals_for_std.append(val)

    def test_shall_continue_warnings(self):
        X, y = load_boston(return_X_y=True)

        inner_fold_length = 7
        my_pipe = Hyperpipe(name='performance_pipe',
                            optimizer='random_search',
                            optimizer_params={'limit_in_minutes': 0.05},
                            metrics=['mean_squared_error'],
                            best_config_metric='mean_squared_error',
                            inner_cv=KFold(n_splits=inner_fold_length),
                            use_test_set=True,
                            project_folder='./tmp',
                            verbosity=0,
                            performance_constraints=[MinimumPerformanceConstraint(strategy='first',
                                                                                  metric='mean_absolute_error',
                                                                                  threshold=50)])

        my_pipe += PipelineElement('StandardScaler')
        my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': IntegerRange(5, 50)})

        with warnings.catch_warnings(record=True) as w:
            my_pipe.fit(X, y)
            assert any("The metric is not calculated." in s for s in [e.message.args[0] for e in w])
