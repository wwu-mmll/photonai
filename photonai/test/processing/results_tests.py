import unittest
import numpy as np

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from photonai.base import Hyperpipe, PipelineElement, OutputSettings


class XPredictor(BaseEstimator, ClassifierMixin):

    _estimator_type = 'classifier'

    def __init__(self):
        self.needs_y = False
        self.needs_covariates = False
        pass

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return X

    def predict_proba(self, X):
        return X/10


# Todo: build dummy estimator predicting X , register it, do hyperpipe analysis, reassure output=input
class ResultHandlerTests(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.y_true = np.linspace(1, 100, 100)
        X = cls.y_true
        cls.inner_fold_nr = 10
        cls.outer_fold_nr = 5

        cls.hyperpipe = Hyperpipe('test_prediction_collection',
                                  inner_cv=KFold(n_splits=cls.inner_fold_nr),
                                  outer_cv=KFold(n_splits=cls.outer_fold_nr),
                                  metrics=['mean_absolute_error', 'mean_squared_error'],
                                  best_config_metric='mean_absolute_error',
                                  output_settings=OutputSettings(save_predictions='all'))

        cls.hyperpipe += PipelineElement('PhotonTestXPredictor')
        cls.hyperpipe.fit(X, cls.y_true)

    def test_get_inner_val_preds(self):
        inner_preds_received = self.hyperpipe.results_handler.get_validation_predictions()
        first_outer_fold_info = next(iter(self.hyperpipe.cross_validation.outer_folds.values()))
        values_to_expect = np.asarray(first_outer_fold_info.train_indices) + 1.0
        self.assertTrue(np.array_equal(inner_preds_received['y_pred'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['y_true'], values_to_expect))
        self.assertTrue(np.array_equal(inner_preds_received['probabilities'], values_to_expect / 10))


    def test_get_outer_val_preds(self):
        outer_preds_received = self.hyperpipe.results_handler.get_test_predictions()
        self.assertTrue(np.array_equal(outer_preds_received['y_pred'], self.y_true))
        self.assertTrue(np.array_equal(outer_preds_received['y_true'], self.y_true))
        self.assertTrue(np.array_equal(outer_preds_received['probabilities'], self.y_true/10))

    def test_write_predictions(self):
        pass


class MDBHelperTests(unittest.TestCase):

    def test_aggregate_metrics_inner_folds(self):
        pass

    def test_aggregate_metrics_outer_folds(self):
        pass
