import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement
from photonai.optimization import IntegerRange, FloatRange, Categorical
from photonai.helper.photon_base_test import PhotonBaseTest
from photonai.helper.helper import XPredictor
from photonai.processing.results_structure import MDBConfig, MDBFoldMetric


class ResultHandlerAndHelperTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(ResultHandlerAndHelperTests, cls).setUpClass()

    def setUp(self):
        super(ResultHandlerAndHelperTests, self).setUp()
        self.inner_fold_nr = 10
        self.outer_fold_nr = 5
        
        self.y_true = np.linspace(1, 100, 100)
        self.X = self.y_true
        
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   outer_cv=KFold(n_splits=self.outer_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   best_config_metric='mean_absolute_error',
                                   project_folder=self.tmp_folder_path,
                                   verbosity=0)

    def test_cv_config_and_dummy_nr(self):
        X, y = load_boston(return_X_y=True)
        self.hyperpipe += PipelineElement('StandardScaler')
        self.hyperpipe += PipelineElement('PCA', {'n_components': IntegerRange(3, 5)})
        self.hyperpipe += PipelineElement('SVR', {'C': FloatRange(0.001, 10, num=5),
                                                  'kernel': Categorical(['linear', 'rbf'])})

        self.hyperpipe.fit(X, y)

        expected_configs = 2 * 5 * 2

        # check version is present
        self.assertIsNotNone(self.hyperpipe.results.version)

        # check nr of outer and inner folds
        self.assertTrue(len(self.hyperpipe.results.outer_folds) == self.outer_fold_nr)
        self.assertTrue(len(self.hyperpipe.cross_validation.outer_folds) == self.outer_fold_nr)

        for outer_fold_id, inner_folds in self.hyperpipe.cross_validation.inner_folds.items():
            self.assertTrue(len(inner_folds) == self.inner_fold_nr)

        for outer_fold_result in self.hyperpipe.results.outer_folds:
            # check that we have the right amount of configs tested in each outer fold
            self.assertTrue(len(outer_fold_result.tested_config_list) == expected_configs)

            for config_result in outer_fold_result.tested_config_list:
                # check that we have the right amount of inner-folds per config
                self.assertTrue(len(config_result.inner_folds) == self.inner_fold_nr)

        self.check_for_dummy()

    def test_get_metric(self):

        metric_list = [MDBFoldMetric(metric_name='a', value=1, operation='raw'),
                       MDBFoldMetric(metric_name='a', value=0.5, operation='mean'),
                       MDBFoldMetric(metric_name='b', value=1, operation='raw'),
                       MDBFoldMetric(metric_name='c', value=1, operation='raw'),
                       MDBFoldMetric(metric_name='c', value=0, operation='mean'),
                       MDBFoldMetric(metric_name='c', value=2, operation='std')]
        doubled_metrics = [MDBFoldMetric(metric_name='a', value=1, operation='raw'),
                           MDBFoldMetric(metric_name='a', value=1, operation='raw')]

        okay_config = MDBConfig()
        okay_config.metrics_test = metric_list
        doubled_config = MDBConfig()
        doubled_config.metrics_test = doubled_metrics

        # raise error when no metric filter infos are given
        with self.assertRaises(ValueError):
            okay_config.get_test_metric(name="", operation="")

        # check doubled metrics
        with self.assertRaises(KeyError):
            doubled_config.get_test_metric(name='a', operation='raw')

        # check None is returned when there is no metric
        self.assertIsNone(okay_config.get_test_metric(name='d', operation='raw'))

        with self.assertRaises(KeyError):
            # b) when there are doubled metrics
            doubled_config.get_test_metric(name='a', operation="raw")

        # check there is "mean" given for when there is no operation
        self.assertEqual(okay_config.get_test_metric(name='a'), 0.5)
        # check there is the correct metric value returned
        self.assertEqual(okay_config.get_test_metric(name='c', operation='std'), 2)

        expected_dict = {'a': 1, 'b': 1, 'c': 1}
        self.assertDictEqual(expected_dict, okay_config.get_test_metric(operation='raw'))

    def check_for_dummy(self):
        self.assertTrue(hasattr(self.hyperpipe.results, 'dummy_estimator'))
        # we should have mean and std for each metric respectively
        expected_dummy_metrics = len(self.hyperpipe.optimization.metrics) * 2
        if self.hyperpipe.cross_validation.use_test_set:
            self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_test) == expected_dummy_metrics)
        # we should have mean and std for each metric respectively
        self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_train) == expected_dummy_metrics)

    def test_get_predictions(self):

        self.hyperpipe += PipelineElement('PhotonTestXPredictor')
        self.hyperpipe.fit(self.X, self.y_true)

        inner_preds_received = self.hyperpipe.results_handler.get_validation_predictions()
        first_outer_fold_info = next(iter(self.hyperpipe.cross_validation.outer_folds.values()))
        values_to_expect = np.asarray(first_outer_fold_info.train_indices) + 1.0
        self.assertTrue(np.array_equal(inner_preds_received['y_pred'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['y_true'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['probabilities'], values_to_expect / 10))

        outer_preds_received = self.hyperpipe.results_handler.get_test_predictions()
        self.assertTrue(np.array_equal(outer_preds_received['y_pred'], self.y_true))
        self.assertTrue(np.array_equal(outer_preds_received['y_true'], self.y_true))
        self.assertTrue(np.array_equal(outer_preds_received['probabilities'], self.y_true / 10))

        csv_file = pd.read_csv(
            os.path.join(self.hyperpipe.output_settings.results_folder, 'best_config_predictions.csv'))
        self.assertTrue(np.array_equal(csv_file.y_pred.values, self.y_true))
        self.assertTrue(np.array_equal(csv_file.y_true.values, self.y_true))
        self.assertTrue(np.array_equal(csv_file.probabilities.values, self.y_true / 10))

    def test_get_predictions_no_outer_cv_eval_final_performance_false(self):
        self.hyperpipe += PipelineElement('PhotonTestXPredictor')
        self.hyperpipe.cross_validation.outer_cv = None
        self.hyperpipe.cross_validation.use_test_set = False
        self.hyperpipe.fit(self.X, self.y_true)
        self.check_predictions_eval_final_performance_false()

    def get_predictions_outer_cv_eval_final_performance_false(self):
        self.hyperpipe += PipelineElement('PhotonTestXPredictor')
        self.hyperpipe.cross_validation.use_test_set = False
        self.hyperpipe.fit(self.X, self.y_true)
        self.check_predictions_eval_final_performance_false()

    def check_predictions_eval_final_performance_false(self):
        inner_preds_received = self.hyperpipe.results_handler.get_validation_predictions()
        first_outer_fold_info = next(iter(self.hyperpipe.cross_validation.outer_folds.values()))
        values_to_expect = np.asarray(first_outer_fold_info.train_indices) + 1.0
        self.assertTrue(np.array_equal(inner_preds_received['y_pred'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['y_true'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['probabilities'], values_to_expect / 10))

        # we are not allowed to evalute the outer_folds test set so we get empty lists here
        outer_fold_predictiosn_received = self.hyperpipe.results_handler.get_test_predictions()
        self.assertTrue(len(outer_fold_predictiosn_received['y_pred']) == 0)
        self.assertTrue(len(outer_fold_predictiosn_received['y_true']) == 0)

        # in case we have no outer cv, we write the inner_cv predictions
        csv_file = pd.read_csv(
            os.path.join(self.hyperpipe.output_settings.results_folder, 'best_config_predictions.csv'))
        self.assertTrue(np.array_equal(csv_file.y_pred.values, values_to_expect))
        self.assertTrue(np.array_equal(csv_file.y_true.values, values_to_expect))
        self.assertTrue(np.array_equal(csv_file.probabilities.values, values_to_expect / 10))

    def test_best_config_stays_the_same(self):
        X, y = load_boston(return_X_y=True)
        self.hyperpipe += PipelineElement('StandardScaler')
        self.hyperpipe += PipelineElement('PCA', {'n_components': [4, 5]}, random_state=42)
        self.hyperpipe += PipelineElement('LinearRegression')
        self.hyperpipe.fit(X, y)

        best_config = self.hyperpipe.results.best_config.config_dict
        expected_best_config = {'PCA__n_components': 5}
        self.assertDictEqual(best_config, expected_best_config)

    def test_metrics_and_aggregations(self):
        
        self.hyperpipe += PipelineElement('PhotonTestXPredictor', change_predictions=True)
        X = np.linspace(0, 99, 100)
        y_true = X
        self.hyperpipe.fit(X, y_true)

        self.metric_assertions()
        self.check_for_dummy()

    def test_metrics_and_aggreation_eval_performance_false(self):
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   use_test_set=False,
                                   best_config_metric='mean_absolute_error',
                                   calculate_metrics_across_folds=True,
                                   project_folder=self.tmp_folder_path)

        self.test_metrics_and_aggregations()

    def test_metrics_and_aggregations_no_outer_cv_but_eval_performance_true(self):
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   outer_cv=KFold(n_splits=self.outer_fold_nr),
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   use_test_set=False,
                                   best_config_metric='mean_absolute_error',
                                   calculate_metrics_per_fold=True,
                                   calculate_metrics_across_folds=True,
                                   project_folder=self.tmp_folder_path)

        self.test_metrics_and_aggregations()

    def metric_assertions(self):
        def check_metrics(metric_name, expected_metric_list, mean_metrics):
            for metric in mean_metrics:
                if metric.metric_name == metric_name:
                    if metric.operation == 'mean':
                        expected_val_mean = np.mean(expected_metric_list)
                        self.assertEqual(expected_val_mean, metric.value)
                    elif metric.operation == 'std':
                        expected_val_std = np.std(expected_metric_list)
                        self.assertAlmostEqual(expected_val_std, metric.value)
            return expected_val_mean, expected_val_std

        outer_collection = {'train': list(), 'test': list()}
        for i, (_, outer_fold) in enumerate(self.hyperpipe.cross_validation.outer_folds.items()):
            outer_fold_results = self.hyperpipe.results.outer_folds[i]
            config = outer_fold_results.tested_config_list[0]
            inner_fold_results = config.inner_folds

            inner_fold_metrics = {'train': list(), 'test': list()}
            for _, inner_fold in self.hyperpipe.cross_validation.inner_folds[outer_fold.fold_id].items():
                tree_result = inner_fold_results[inner_fold.fold_nr - 1]

                global_test_indices = outer_fold.train_indices[inner_fold.test_indices]
                expected_test_mae = mean_absolute_error(XPredictor.adapt_X(global_test_indices),
                                                        global_test_indices)
                inner_fold_metrics['test'].append(expected_test_mae)
                self.assertEqual(expected_test_mae, tree_result.validation.metrics['mean_absolute_error'])
                self.assertTrue(np.array_equal(tree_result.validation.indices, inner_fold.test_indices))
                self.assertEqual(len(global_test_indices), len(tree_result.validation.y_true))
                self.assertEqual(len(global_test_indices), len(tree_result.validation.y_pred))

                global_train_indices = outer_fold.train_indices[inner_fold.train_indices]
                expected_train_mae = mean_absolute_error(XPredictor.adapt_X(global_train_indices),
                                                         global_train_indices)
                inner_fold_metrics['train'].append(expected_train_mae)
                self.assertEqual(expected_train_mae, tree_result.training.metrics['mean_absolute_error'])
                # check that indices are as expected and the right number of y_pred and y_true exist in the tree
                self.assertTrue(np.array_equal(tree_result.training.indices, inner_fold.train_indices))
                self.assertEqual(len(global_train_indices), len(tree_result.training.y_true))
                self.assertEqual(len(global_train_indices), len(tree_result.training.y_pred))

                # get expected train and test mean and std respectively and calculate mean and std again.

            check_metrics('mean_absolute_error', inner_fold_metrics['train'], config.metrics_train)
            check_metrics('mean_absolute_error', inner_fold_metrics['test'], config.metrics_test)

            # calculate metrics across folds
            if self.hyperpipe.cross_validation.calculate_metrics_across_folds:
                expected_mean_absolute_error_across_folds = mean_absolute_error(XPredictor.adapt_X(outer_fold.train_indices),
                                                                                outer_fold.train_indices)
                actual_mean_absolute_error_across_folds = config.get_train_metric('mean_absolute_error', "raw")
                self.assertEqual(expected_mean_absolute_error_across_folds, actual_mean_absolute_error_across_folds)

            if self.hyperpipe.cross_validation.use_test_set:
                expected_outer_test_mae = mean_absolute_error(XPredictor.adapt_X(outer_fold.test_indices),
                                                              outer_fold.test_indices)

                self.assertTrue(np.array_equal(outer_fold_results.best_config.best_config_score.validation.indices,
                                         outer_fold.test_indices))
                self.assertEqual(len(outer_fold.test_indices),
                                 len(outer_fold_results.best_config.best_config_score.validation.y_true))
                self.assertEqual(len(outer_fold.test_indices),
                                 len(outer_fold_results.best_config.best_config_score.validation.y_pred))

                # check that indices are as expected and the right number of y_pred and y_true exist in the tree
                self.assertTrue(np.array_equal(outer_fold_results.best_config.best_config_score.training.indices,
                                               outer_fold.train_indices))
                self.assertEqual(len(outer_fold.train_indices),
                                 len(outer_fold_results.best_config.best_config_score.training.y_true))
                self.assertEqual(len(outer_fold.train_indices),
                                 len(outer_fold_results.best_config.best_config_score.training.y_pred))
            else:
                # if we dont use the test set, we want the values from the inner_cv to be copied
                expected_outer_test_mae = outer_fold_results.best_config.get_test_metric('mean_absolute_error', 'mean')

                self.assertTrue(outer_fold_results.best_config.best_config_score.validation.metrics_copied_from_inner)
                self.assertTrue(outer_fold_results.best_config.best_config_score.training.metrics_copied_from_inner)

            outer_collection['test'].append(expected_outer_test_mae)
            self.assertEqual(outer_fold_results.best_config.best_config_score.validation.metrics['mean_absolute_error'],
                             expected_outer_test_mae)

            expected_outer_train_mae = mean_absolute_error(XPredictor.adapt_X(outer_fold.train_indices),
                                                           outer_fold.train_indices)
            outer_collection['train'].append(expected_outer_train_mae)
            self.assertAlmostEqual(outer_fold_results.best_config.best_config_score.training.metrics['mean_absolute_error'],
                                   expected_outer_train_mae)

        # check again in overall best config attribute
        check_metrics('mean_absolute_error', outer_collection['train'],
                      self.hyperpipe.results.metrics_train)

        check_metrics('mean_absolute_error', outer_collection['test'],
                      self.hyperpipe.results.metrics_test)

        # check if those agree with helper function output
        outer_fold_performances = self.hyperpipe.results_handler.get_performance_outer_folds()
        self.assertListEqual(outer_fold_performances['mean_absolute_error'], outer_collection['test'])

    def test_three_levels_of_feature_importances(self):
        hyperpipe = Hyperpipe('fimps',
                              inner_cv=KFold(n_splits=4),
                              outer_cv=KFold(n_splits=3),
                              metrics=['mean_absolute_error', 'mean_squared_error'],
                              best_config_metric='mean_squared_error',
                              project_folder=self.tmp_folder_path)
        hyperpipe += PipelineElement('StandardScaler')
        hyperpipe += PipelineElement('DecisionTreeRegressor')
        X, y = load_boston(return_X_y=True)
        hyperpipe.fit(X, y)

        exepcted_nr_of_feature_importances = X.shape[1]
        self.assertTrue(len(hyperpipe.results.best_config_feature_importances) == exepcted_nr_of_feature_importances)

        for outer_fold in hyperpipe.results.outer_folds:
            self.assertTrue(len(outer_fold.best_config.best_config_score.feature_importances) == exepcted_nr_of_feature_importances)
            for inner_fold in outer_fold.best_config.inner_folds:
                self.assertTrue(len(inner_fold.feature_importances) == exepcted_nr_of_feature_importances)
