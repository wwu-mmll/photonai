import unittest

import numpy as np
from sklearn.model_selection import KFold, GroupShuffleSplit

from photonai.processing.photon_folds import FoldInfo


class PhotonFoldsTests(unittest.TestCase):

    def setUp(self):
        self.num_subjects = 100
        self.X = np.random.random((self.num_subjects, 45))
        self.y = np.random.randn(self.num_subjects)
        groups = []
        [groups.extend((np.ones((10,)) * i).tolist()) for i in range(10)]
        self.kwargs = {'groups': np.asarray(groups)}

    def base_assertions(self, cv, nr_of_folds, eval_final_performance=True):
        fold_list = FoldInfo.generate_folds(cv, self.X, self.y, self.kwargs,
                                            eval_final_performance=eval_final_performance)
        self.assertTrue(len(fold_list) == nr_of_folds)
        for generated_fold in fold_list:
            self.assertEqual(len(generated_fold.train_indices), (nr_of_folds - 1) * (self.num_subjects / nr_of_folds))
            self.assertEqual(len(generated_fold.test_indices), (self.num_subjects / nr_of_folds))
        # we always start with 1 for the investigator
        self.assertEqual(fold_list[0].fold_nr, 1)
        return fold_list

    def test_k_fold_generation(self):
        nr_of_folds = 5
        cv = KFold(n_splits=nr_of_folds)
        self.base_assertions(cv, nr_of_folds)

    def test_group_k_fold_generation(self):
        nr_of_folds = 5
        cv = GroupShuffleSplit(n_splits=nr_of_folds)
        fold_list = self.base_assertions(cv, nr_of_folds)

        # test that we have used the groups from the kwargs
        for generated_fold in fold_list:
            train_groups = np.unique(self.kwargs['groups'][generated_fold.train_indices])
            test_groups = np.unique(self.kwargs['groups'][generated_fold.test_indices])
            self.assertFalse(bool(set(train_groups) & set(test_groups)))

    def test_no_cv_strategy(self):
        test_size = 0.15
        fold_list = FoldInfo.generate_folds(None, self.X, self.y, self.kwargs, eval_final_performance=True,
                                            test_size=test_size)
        # check that we have only one outer fold, that is split in training and test according to test size
        self.assertEqual(len(fold_list), 1)
        self.assertEqual(len(fold_list[0].train_indices), (1 - test_size) * self.num_subjects)
        self.assertEqual(len(fold_list[0].test_indices), test_size * self.num_subjects)

    def test_no_cv_strategy_eval_final_performance_false(self):
        test_size = 0.15
        fold_list = FoldInfo.generate_folds(None, self.X, self.y, self.kwargs, eval_final_performance=False,
                                            test_size=test_size)
        # check that we have only one outer fold, that is split in training and test according to test size
        self.assertEqual(len(fold_list), 1)
        self.assertEqual(len(fold_list[0].train_indices), self.num_subjects)
        self.assertEqual(len(fold_list[0].test_indices), 0)

        fold_list = FoldInfo.generate_folds(None, range(len(self.y)), self.y, self.kwargs, eval_final_performance=False)
        self.assertEqual(len(fold_list), 1)
        self.assertTrue(np.array_equal(range(len(self.y)), fold_list[0].train_indices))

    def test_k_fold_generation_eval_final_performance_false(self):
        # this should change nothing
        nr_of_folds = 10
        cv = KFold(n_splits=nr_of_folds)
        self.base_assertions(cv, nr_of_folds, eval_final_performance=False)

    def test_data_overview(self):
        expected_outcome = {str(i): 10 for i in range(10)}
        data_count = FoldInfo.data_overview(self.kwargs['groups'].astype(int))
        self.assertDictEqual(expected_outcome, data_count)
