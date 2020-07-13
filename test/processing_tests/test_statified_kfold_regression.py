import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer

from photonai.processing.cross_validation import StratifiedKFoldRegression
from photonai.processing.photon_folds import FoldInfo


class StratifiedRegressionTest(unittest.TestCase):

    def setUp(self):
        # set up all we need
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.kwargs = {'groups': np.random.randint(-10, 10, (self.y.shape[0]))}
        self.outer_cv = StratifiedKFoldRegression(n_splits=2, random_state=15, shuffle=True)
        self.inner_cv = StratifiedKFoldRegression(n_splits=2, random_state=15, shuffle=True)
        self.outer_folds = FoldInfo.generate_folds(self.outer_cv, self.X, self.y, {})

        # it's important to say at this point, that "groups" is a giant misnomer for StratifiedKFoldRegression
        # if there is a groups variable in kwargs, PHOTON will forward this variable to StatifiedKFoldRegression
        # however, this variable shouldn't be a group but a continuous variable
        self.outer_folds_w_groups = FoldInfo.generate_folds(self.outer_cv, self.X, self.y, self.kwargs)

    def test_stratified_regression(self):
        splits = list()
        splits_pipe = list()
        for train, test in self.outer_cv.split(self.X, self.y):
            splits.append(train)
            splits.append(test)

        for fold in self.outer_folds:
            splits_pipe.append(fold.train_indices)
            splits_pipe.append(fold.test_indices)

        for i in range(len(splits)):
            np.testing.assert_array_equal(splits[i], splits_pipe[i])

    def test_stratified_regression_w_groups(self):
        splits = list()
        splits_pipe = list()
        for train, test in self.outer_cv.split(self.X, self.kwargs['groups']):
            splits.append(train)
            splits.append(test)

        for fold in self.outer_folds_w_groups:
            splits_pipe.append(fold.train_indices)
            splits_pipe.append(fold.test_indices)

        for i in range(len(splits)):
            np.testing.assert_array_equal(splits[i], splits_pipe[i])

    def test_multi_dimensional_y(self):
        y = np.random.randint(-10, 10, (self.y.shape[0], 2))
        with self.assertRaises(NotImplementedError):
            FoldInfo.generate_folds(self.outer_cv, self.X, y, {})
