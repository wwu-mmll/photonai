import unittest
import numpy as np
from sklearn.model_selection import KFold


# Todo: build dummy estimator predicting X , register it, do hyperpipe analysis, reassure output=input
class ResultHandlerTests(unittest.TestCase):

    def setUp(self):
        y_true = np.linspace(1, 101, 100)
        fold_nr = 10


    def test_get_inner_val_preds(self):
        pass

    def test_get_outer_val_preds(self):
        pass

    def test_write_predictions(self):
        pass


class MDBHelperTests(unittest.TestCase):

    def test_aggregate_metrics_inner_folds(self):
        pass

    def test_aggregate_metrics_outer_folds(self):
        pass
