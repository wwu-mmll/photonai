import uuid
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold, \
    StratifiedShuffleSplit, ShuffleSplit

from photonai.photonlogger.logger import logger
from photonai.processing.cross_validation import StratifiedKFoldRegression


class FoldInfo:
    """Fold Info.

    Provides all information of a fold.
    This guarantees the mapping between ID,
    number and indices for training and testing.

    Parameters
    ----------
    fold_id: default=None
        UUID as a unique indicator.

    fold_nr: int, default=0
        Position of Fold in cross-validation.

    train_indices: list, default=None
        List of all indices defined in the training set.

    test_indices: list, default=None
        List of all indices defined in the test set.

    """

    def __init__(self, fold_id=None, fold_nr: int = 0, train_indices: list = None, test_indices: list = None):
        self.fold_id = fold_id
        self.fold_nr = fold_nr
        self.train_indices = train_indices
        self.test_indices = test_indices

    @staticmethod
    def data_overview(y):
        if len(y.shape) > 1:
            # one hot encoded
            logger.info("One Hot Encoded data fold information not yet implemented")
            return {}
        else:
            unique, counts = np.unique(y, return_counts=True)
            # replacing is necessary for float inputs, because mongoDB does not allow '.'-char
            unique = [str(u).replace(".", "_") for u in unique]
            counts = [int(c) for c in counts]
            return dict(zip(unique, counts))

    @staticmethod
    def generate_folds(cv_strategy, X, y, kwargs, eval_final_performance=True, test_size=0.2):
        """Generates the training and  test set indices for the hyperparameter search.
        Returns a tuple of training and test indices.

        Parameters
        ----------
        cv_strategy
            Strategy for generating train/test sets.

        X: ndarray
            Data X to split.

        y: ndarray
            Targets to split.

        kwargs: dict
            Mostly empty. Responds only to the keyword "groups"
            and returns the corresponding value in the cv_strategy.

        eval_final_performance: bool, default=True
            See below in notes and code for the explanation.

        test_size: float, default=0.2,
            Only relevant if the final performance is performed.
            Defines the percentage size of the test set.

        Notes
        -----
        - If there is a strategy given for the outer cross validation the strategy is called to split the data
            - additionally, if a group variable and a GroupCV is passed, split data according to groups.
            - if a group variable and a StratifiedCV is passed, split data according to groups and ignore targets when
            stratifying the data.
            - if no group variable but a StratifiedCV is passed, split data according to targets.
        - If no strategy is given and eval_final_performance is True, all data is used for training.
        - If no strategy is given and eval_final_performance is False: a test set is seperated from the
          training and validation set by the parameter test_size with ShuffleSplit.

        """
        # if there is a CV Object for cross validating the hyperparameter search
        if 'groups' in kwargs.keys():
            groups = kwargs['groups']
        else:
            groups = None

        if cv_strategy is not None:
            if groups is not None and (isinstance(cv_strategy, (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut))):
                try:
                    data_test_cases = cv_strategy.split(X, y, groups)
                except:
                    logger.error("Could not split data according to groups")
            elif groups is not None and (isinstance(cv_strategy, (StratifiedKFoldRegression,
                                                                  StratifiedKFold,
                                                                  StratifiedShuffleSplit))):
                try:
                    data_test_cases = cv_strategy.split(X, groups)
                except:
                    logger.error("Could not stratify data for outer cross validation according to "
                                 "group variable")
            else:
                data_test_cases = cv_strategy.split(X, y)

        # in case we do not want to divide between validation and test set
        # Re eval_final_performance:
        #         # set eval_final_performance to False because
        #         # 1. if no cv-object is given, no split is performed --> seems more logical
        #         #    than passing nothing, passing no cv-object but getting
        #         #    an 80/20 split by default
        #         # 2. if cv-object is given, split is performed but we don't peek
        #         #    into the test set --> thus we can evaluate more hp configs
        #         #    later without double dipping
        elif not eval_final_performance:
            data_test_cases = FoldInfo._yield_all_data(X)
        # the default is dividing one time into a validation and test set
        else:
            train_test_cv_object = ShuffleSplit(n_splits=1, test_size=test_size)
            data_test_cases = train_test_cv_object.split(X, y)

        fold_objects = list()
        for i, (train_indices, test_indices) in enumerate(data_test_cases):
            fold_info_obj = FoldInfo(fold_id=uuid.uuid4(),
                                     fold_nr=i+1,
                                     train_indices=train_indices,
                                     test_indices=test_indices)
            fold_objects.append(fold_info_obj)

        return fold_objects

    @staticmethod
    def _yield_all_data(X):
        """
        Helper function that iteratively returns the data stored in self.X
        Returns an iterable version of self.X
        """
        if hasattr(X, 'shape'):
            yield np.asarray(list(range(X.shape[0]))), []
        else:
            yield np.asarray(list(range(len(X)))), []
