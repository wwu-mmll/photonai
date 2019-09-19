import random
import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer

from photonai.helper.helper import PhotonDataHelper
from photonai.processing.photon_folds import FoldInfo


class FoldInfoTests(unittest.TestCase):

    def setUp(self):
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_class_distribution_info(self):
        unique, counts = np.unique(self.__y, return_counts=True)
        nr_dict = FoldInfo.data_overview(self.__y)
        self.assertEqual(counts[1], nr_dict['1'])


class DataHelperTests(unittest.TestCase):

    def test_data_split_indices(self):
        vals = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
        vals_str = np.array([ascii(i) for i in vals])
        random_features = np.random.randn(10, 20)
        kwargs = {'test': vals,
                  'subtest': vals_str,
                  'random': random_features}
        pick_list = [1, 3, 5]
        splitted_X, splitted_y, splitted_example = PhotonDataHelper.split_data(random_features, vals,
                                                   kwargs, indices=pick_list)
        self.assertTrue(np.array_equal(splitted_X, random_features[pick_list]))
        self.assertTrue(np.array_equal(splitted_y, vals[pick_list]))
        self.assertTrue(np.array_equal(splitted_example['test'], vals[pick_list]))
        self.assertTrue(np.array_equal(splitted_example['subtest'], vals_str[pick_list]))
        self.assertTrue(np.array_equal(splitted_example['random'], random_features[pick_list]))

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
        joined_idx = PhotonDataHelper.stack_data_vertically(idx_list_two, idx_list_one)
        self.assertTrue(np.array_equal(X_new, X[joined_idx]))
        self.assertTrue(np.array_equal(y_new, y[joined_idx]))
        self.assertTrue(np.array_equal(kwargs_new['test'], kwargs['test'][joined_idx]))

        # now resort and see if that works too
        X_resorted, y_resorted, kwargs_resorted = PhotonDataHelper.resort_splitted_data(X_new, y_new, kwargs_new, joined_idx)
        self.assertTrue(np.array_equal(X_resorted, X))
        self.assertTrue(np.array_equal(y_resorted, y))
        self.assertListEqual(list(kwargs_resorted.keys()), list(kwargs.keys()))
        self.assertTrue(np.array_equal(kwargs_resorted['test'], kwargs['test']))

    def test_concatenate_dict(self):
        dict_a = {'variable_one': np.random.randn(10),
                  'variable_two': np.random.randn(15)}
        dict_b = {'variable_one': np.random.randn(20),
                  'variable_two': np.random.randn(20)}
        dict_c = {'variable_one': np.random.randn(10, 10),
                  'variable_two': np.random.randn(15, 15)}
        dict_d = {'variable_one': np.random.randn(20, 10),
                  'variable_two': np.random.randn(20, 15)}
        dict_e = {}

        dict_a_b = PhotonDataHelper.join_dictionaries(dict_a, dict_b)
        dict_c_d = PhotonDataHelper.join_dictionaries(dict_c, dict_d)
        dict_e_a = PhotonDataHelper.join_dictionaries(dict_e, dict_a)
        self.assertEqual(len(dict_a_b['variable_one']), 30)
        self.assertEqual(len(dict_a_b['variable_two']), 35)
        self.assertEqual(dict_c_d['variable_one'].shape, (30, 10))
        self.assertEqual(dict_c_d['variable_two'].shape, (35, 15))
        self.assertEqual(len(dict_e_a['variable_one']), 10)
        self.assertEqual(len(dict_e_a['variable_two']), 15)

    def test_index_dict(self):
        labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dict_a = {'variable_one': np.random.randn(10),
                  'variable_two': np.random.randn(10, 10)}
        dict_a_1 = PhotonDataHelper.index_dict(dict_a, labels == 0)
        dict_a_2 = PhotonDataHelper.index_dict(dict_a, labels == 1)
        self.assertEqual(len(dict_a_1['variable_one']), 5)
        self.assertEqual(dict_a_2['variable_two'].shape, (5, 10))
