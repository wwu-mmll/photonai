from sklearn.utils import check_random_state
from sklearn.model_selection._split import _BaseKFold
import numpy as np

from itertools import zip_longest


class StratifiedKFoldRegression(_BaseKFold):
    """Stratified K-Folds cross-validator for continuous target values (regression tasks)
       Provides train/test indices to split data in train/test sets.
       This cross-validation object is a variation of StratifiedKFold that returns
       stratified folds with regard to a continuous target. The folds are made by first sorting
       samples with respect to y. Subsequently, we step through each consecutive k samples and
       randomly allocate exactly one of them to one of the test folds. For a detailed description
       of the process see: http://scottclowe.com/2016-03-19-stratified-regression-partitions/.
       Read more in the :ref:`User Guide <cross_validation>`.
       Parameters
       ----------
       n_splits : int, default=3
           Number of folds. Must be at least 2.
       shuffle : boolean, optional
           Whether to shuffle each stratification of the data before splitting
           into batches.
       random_state : int, RandomState instance or None, optional, default=None
           If int, random_state is the seed used by the random number generator;
           If RandomState instance, random_state is the random number generator;
           If None, the random number generator is the RandomState instance used
           by `np.random`. Used when ``shuffle`` == True.
       Examples
       --------
       >>> from photonai.validation.cross_validation import StratifiedKFoldRegression
       >>> X = np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
       >>> y = np.array([0, 1, 2, 3, 4, 5])
       >>> skf = StratifiedKFoldRegression(n_splits=2)
       >>> skf.get_n_splits(X, y)
       2
       >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
       StratifiedKFoldRegression(n_splits=2, random_state=None, shuffle=False)
       >>> for train_index, test_index in skf.split(X, y):
       ...    print("TRAIN:", train_index, "TEST:", test_index)
       ...    X_train, X_test = X[train_index], X[test_index]
       ...    y_train, y_test = y[train_index], y[test_index]
       TRAIN: [1 3 5] TEST: [0 2 4]
       TRAIN: [0 2 4] TEST: [1 3 5]
       Notes
       -----
       All the folds have size ``trunc(n_samples / n_splits)``, the last one has
       the complementary.
       """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedKFoldRegression, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None):
        rng = self.random_state
        n_splits = self.n_splits
        y = np.asarray(y)
        if y.ndim > 1:
            raise ValueError('Target data has more than one dimension. Must be single vector of continuous values.')
        n_samples = y.shape[0]
        self.n_samples = n_samples
        sort_indices = np.argsort(y)
        min_test_samples_per_fold = int(np.ceil(n_samples / n_splits))

        test_folds = [[] for _ in range(n_splits)]
        current = 0
        for i in range(min_test_samples_per_fold):
            start, stop = current, current + n_splits
            if i+1 < min_test_samples_per_fold:
                subset = sort_indices[start:stop]
            else:
                subset = sort_indices[start:]

            if self.shuffle:
                check_random_state(rng).shuffle(subset)
            for k in range(len(subset)):
                test_folds[k].append(subset[k])
            current = stop
        return test_folds


    def _iter_test_masks(self, X, y, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            test_mask = np.zeros(self.n_samples, dtype=np.bool)
            test_mask[test_folds[i]] = True
            yield test_mask

    def split(self, X, y, groups=None):
        return super(StratifiedKFoldRegression, self).split(X, y, groups)



class OutlierKFold(_BaseKFold):

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(OutlierKFold, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None):

        rng = self.random_state
        n_splits = self.n_splits
        y = np.asarray(y)
        if y.ndim > 1:
            raise ValueError('Target data has more than one dimension. Must be single vector of continuous values.')
        n_samples = y.shape[0]
        self.n_samples = n_samples

        one_class = np.where(y == 1)[0]
        outlier = np.where(y == -1)[0]

        one_class_chunks = self.chunk_array(one_class)
        outlier_chunks = self.chunk_array(outlier)

        folds = [np.concatenate((i, j)) for i, j in zip(one_class_chunks, outlier_chunks)]

        return folds

    def chunk_array(self, index_list: list):

        num_samples = len(index_list)
        test_samples_per_fold = int(np.ceil(num_samples / self.n_splits))
        test_folds = []

        for i in range(0, num_samples, test_samples_per_fold):
            stop = i+test_samples_per_fold
            if stop > num_samples:
                stop = num_samples
            test_folds.append(index_list[i:stop])
        return test_folds

    def _iter_test_masks(self, X, y, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            test_mask = np.zeros(self.n_samples, dtype=np.bool)
            test_mask[test_folds[i]] = True
            yield test_mask

    def split(self, X, y, groups=None):
        return super(OutlierKFold, self).split(X, y, groups)




