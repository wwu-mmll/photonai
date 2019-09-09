import unittest
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, ClassifierMixin
from photonai.base import Hyperpipe, PipelineElement, OutputSettings


class XPredictor(BaseEstimator, ClassifierMixin):

    _estimator_type = 'classifier'

    def __init__(self, change_predictions = False):
        self.needs_y = False
        self.needs_covariates = False
        self.change_predictions = change_predictions
        pass

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, **kwargs):
        if self.change_predictions:
            # change it relative to value so that it is fold-specific
            return XPredictor.adapt_X(X)
        return X

    @staticmethod
    def adapt_X(X):
        return [i-(0.1*i) for i in X]

    def predict_proba(self, X):
        return X/10


# Todo: build dummy estimator predicting X , register it, do hyperpipe analysis, reassure output=input
class ResultHandlerAndHelperTests(unittest.TestCase):

    def setUp(self):

        self.inner_fold_nr = 10
        self.outer_fold_nr = 5
        
        self.y_true = np.linspace(1, 100, 100)
        self.X = self.y_true
        
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   outer_cv=KFold(n_splits=self.outer_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   best_config_metric='mean_absolute_error',
                                   output_settings=OutputSettings(save_predictions='all',
                                                                  project_folder='./tmp'))

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
        self.assertTrue(np.array_equal(outer_preds_received['probabilities'], self.y_true/10))

        csv_file = pd.read_csv(os.path.join(self.hyperpipe.output_settings.results_folder, 'best_config_predictions.csv'))
        self.assertTrue(np.array_equal(csv_file.y_pred.values, self.y_true))
        self.assertTrue(np.array_equal(csv_file.y_true.values, self.y_true))
        self.assertTrue(np.array_equal(csv_file.probabilities.values, self.y_true / 10))
    
    def test_metrics_and_aggregations(self):
        
        self.hyperpipe += PipelineElement('PhotonTestXPredictor', change_predictions=True)
        X = np.linspace(0, 99, 100)
        y_true = X
        self.hyperpipe.fit(X, y_true)

        self.metric_assertions()

    def test_metrics_and_aggreation_eval_performance_false(self):
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   eval_final_performance=False,
                                   best_config_metric='mean_absolute_error',
                                   output_settings=OutputSettings(save_predictions='all',
                                                                  project_folder='./tmp'))

        self.test_metrics_and_aggregations()

    def test_metrics_and_aggregations_no_outer_cv_but_eval_performance_true(self):
        self.hyperpipe = Hyperpipe('test_prediction_collection',
                                   outer_cv=KFold(n_splits=self.outer_fold_nr),
                                   inner_cv=KFold(n_splits=self.inner_fold_nr),
                                   metrics=['mean_absolute_error', 'mean_squared_error'],
                                   eval_final_performance=False,
                                   best_config_metric='mean_absolute_error',
                                   output_settings=OutputSettings(save_predictions='all',
                                                                  project_folder='./tmp'))

        self.test_metrics_and_aggregations()

    def metric_assertions(self):
        def check_metrics(metric_name, expected_metric_list, mean_metrics):
            for metric in mean_metrics:
                if metric.metric_name == metric_name:
                    if metric.operation == 'FoldOperations.MEAN':
                        expected_val_mean = np.mean(expected_metric_list)
                        self.assertEqual(expected_val_mean, metric.value)
                    elif metric.operation == 'FoldOperations.STD':
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

                test_indices_to_check = outer_fold.train_indices[inner_fold.test_indices]
                expected_test_mae = mean_absolute_error(XPredictor.adapt_X(test_indices_to_check),
                                                        test_indices_to_check)
                inner_fold_metrics['test'].append(expected_test_mae)
                self.assertEqual(expected_test_mae, tree_result.validation.metrics['mean_absolute_error'])

                train_indices_to_check = outer_fold.train_indices[inner_fold.train_indices]
                expected_train_mae = mean_absolute_error(XPredictor.adapt_X(train_indices_to_check),
                                                         train_indices_to_check)
                inner_fold_metrics['train'].append(expected_train_mae)
                self.assertEqual(expected_train_mae, tree_result.training.metrics['mean_absolute_error'])

                # get expected train and test mean and std respectively and calculate mean and std again.

            check_metrics('mean_absolute_error', inner_fold_metrics['train'], config.metrics_train)
            check_metrics('mean_absolute_error', inner_fold_metrics['test'], config.metrics_test)

            if self.hyperpipe.cross_validation.eval_final_performance:
                expected_outer_test_mae = mean_absolute_error(XPredictor.adapt_X(outer_fold.test_indices),
                                                              outer_fold.test_indices)
            else:
                # if we dont use the test set, we want the values from the inner_cv to be copied
                expected_outer_test_mae = [m.value for m in outer_fold_results.best_config.metrics_test
                                           if m.metric_name == 'mean_absolute_error'
                                           and m.operation == 'FoldOperations.MEAN']
                if len(expected_outer_test_mae)>0:
                    expected_outer_test_mae = expected_outer_test_mae[0]

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

