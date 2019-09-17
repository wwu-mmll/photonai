import unittest
import random
import numpy as np

from sklearn.datasets import load_breast_cancer

from photonai.processing.photon_folds import FoldInfo
from photonai.helper.helper import PhotonDataHelper


class FoldInfoTests(unittest.TestCase):

    def setUp(self):
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_class_distribution_info(self):
        unique, counts = np.unique(self.__y, return_counts=True)
        nr_dict = FoldInfo._data_overview(self.__y)
        self.assertEqual(counts[1], nr_dict['1'])


class DataHelperTests(unittest.TestCase):

    def test_split_join_resorting(self):
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        kwargs = {'test': np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])}

        X_new, y_new, kwargs_new = list(), list(), dict()

        # first randomly split the data and append them to X_new, y_new, kwargs_new
        idx_list_one, idx_list_two = list(), list()
        for idx in range(len(X)):
            if bool(random.getrandbits(1)):
                idx_list_one.append(idx)
            else:
                idx_list_two.append(idx)

        for ilist in [idx_list_two, idx_list_one]:
            for idx in ilist:

                X_batched, y_batched, kwargs_batched = PhotonDataHelper.split_data(X, y, kwargs, idx, idx)

                # test if batching works
                self.assertEqual(X_batched, X[idx])
                self.assertEqual(y_batched, y[idx])
                self.assertDictEqual(kwargs_batched, {'test': [kwargs['test'][idx]]})

                # then join again
                X_new, y_new, kwargs_new = PhotonDataHelper.join_data(X_new, X_batched, y_new, y_batched, kwargs_new, kwargs_batched)

        # test if joining works
        joined_idx = PhotonDataHelper.stack_results(idx_list_one, idx_list_two)
        self.assertTrue(np.array_equal(X_new, X[joined_idx]))
        self.assertTrue(np.array_equal(y_new, y[joined_idx]))
        self.assertTrue(np.array_equal(kwargs_new['test'], kwargs['test'][joined_idx]))

        # now resort and see if that works too
        X_resorted, y_resorted, kwargs_resorted = PhotonDataHelper.resort_splitted_data(X_new, y_new, kwargs_new, joined_idx)
        self.assertTrue(np.array_equal(X_resorted, X))
        self.assertTrue(np.array_equal(y_resorted, y))
        self.assertListEqual(list(kwargs_resorted.keys()), list(kwargs.keys()))
        self.assertTrue(np.array_equal(kwargs_resorted['test'], kwargs['test']))


