"""
Define custom metrics here
The method stub of all metrics is
function_name(y_true, y_pred)
"""

import numpy as np
from scipy.stats import spearmanr
from photonai.photonlogger.logger import logger

from sklearn.metrics import accuracy_score


class Scorer(object):
    """
    Transforms a string literal into an callable instance of a particular metric
    """

    ELEMENT_DICTIONARY = {
        # Classification
        'matthews_corrcoef': ('sklearn.metrics', 'matthews_corrcoef', 'score'),
        'accuracy': ('sklearn.metrics', 'accuracy_score', 'score'),
        'f1_score': ('sklearn.metrics', 'f1_score', 'score'),
        'hamming_loss': ('sklearn.metrics', 'hamming_loss', 'error'),
        'log_loss': ('sklearn.metrics', 'log_loss', 'error'),
        'precision': ('sklearn.metrics', 'precision_score', 'score'),
        'recall': ('sklearn.metrics', 'recall_score', 'score'),
        'auc': ('sklearn.metrics', 'roc_auc_score', 'score'),
        'sensitivity': ('photonai.processing.metrics', 'sensitivity', 'score'),
        'specificity': ('photonai.processing.metrics', 'specificity', 'score'),
        'balanced_accuracy': ('photonai.processing.metrics', 'balanced_accuracy', 'score'),
        'categorical_accuracy': ('photonai.processing.metrics', 'categorical_accuracy_score', 'score'),

        # Regression
        'mean_squared_error': ('sklearn.metrics', 'mean_squared_error', 'error'),
        'mean_absolute_error': ('sklearn.metrics', 'mean_absolute_error', 'error'),
        'explained_variance': ('sklearn.metrics', 'explained_variance_score', 'score'),
        'r2': ('sklearn.metrics', 'r2_score', 'score'),
        'pearson_correlation': ('photonai.processing.metrics', 'pearson_correlation', 'score'),
        'spearman_correlation': ('photonai.processing.metrics', 'spearman_correlation', 'score'),
        'variance_explained':  ('photonai.processing.metrics', 'variance_explained_score', 'score')

    }

    @classmethod
    def create(cls, metric):
        """
        Searches for the metric by name and instantiates the according calculation function
        :param metric: the name of the metric as encoded in the ELEMENT_DICTIONARY
        :type metric: str
        :return: a callable instance of the metric calculation
        """
        if metric in Scorer.ELEMENT_DICTIONARY:
            try:
                desired_class_info = Scorer.ELEMENT_DICTIONARY[metric]
                desired_class_home = desired_class_info[0]
                desired_class_name = desired_class_info[1]
                imported_module = __import__(desired_class_home, globals(),
                                             locals(), desired_class_name, 0)
                desired_class = getattr(imported_module, desired_class_name)
                scoring_method = desired_class
                return scoring_method
            except AttributeError as ae:
                logger.error('ValueError: Could not find according class: '
                               + Scorer.ELEMENT_DICTIONARY[metric])
                raise ValueError('Could not find according class:',
                                 Scorer.ELEMENT_DICTIONARY[metric])
        else:
            logger.error('NameError: Metric not supported right now:' + metric)
            # raise Warning('Metric not supported right now:', metric)
            return None

    @staticmethod
    def greater_is_better_distinction(metric):
        if metric in Scorer.ELEMENT_DICTIONARY:
            # for now do a simple hack and set greater_is_better
            # by looking at error/score in metric name
            specifier = Scorer.ELEMENT_DICTIONARY[metric][2]
            if specifier == 'score':
                return True
            elif specifier == 'error':
                return False
            else:
                # Todo: better error checking?
                error_msg = "Metric not suitable for optimizer."
                logger.error(error_msg)
                raise NameError(error_msg)
        else:
            logger.error('Specify valid metric to choose best config.')
        raise NameError('Specify valid metric to choose best config.')

    @staticmethod
    def calculate_metrics(y_true, y_pred, metrics):
        """
        Applies all metrics to the given predicted and true values.
        The metrics are encoded via a string literal which is mapped to the according calculation function
        :param y_true: the truth values
        :type y_true: list
        :param y_pred: the predicted values
        :param metrics: list
        :return: dict of metrics
        """

        # Todo: HOW TO CHECK IF ITS REGRESSION?!
        # The following works only for classification
        # if np.ndim(y_pred) == 2:
        #     y_pred = one_hot_to_binary(y_pred)
        #     logger.warn("test_predictions was one hot encoded => transformed to binary")
        #
        # if np.ndim(y_true) == 2:
        #     y_true = one_hot_to_binary(y_true)
        #     logger.warn("test_y was one hot encoded => transformed to binary")

        output_metrics = {}
        if metrics:
            for metric in metrics:
                scorer = Scorer.create(metric)
                if scorer is not None:
                    scorer_value = scorer(y_true, y_pred)
                    logger.debug(str(scorer_value))
                    output_metrics[metric] = scorer_value
                else:
                    output_metrics[metric] = np.nan

        return output_metrics


def binary_to_one_hot(binary_vector):
    classes = np.unique(binary_vector)
    out = np.zeros((binary_vector.shape[0], len(classes)),  dtype=np.int)
    for i, c in enumerate(classes):
        out[binary_vector == c, i] = 1
    return out


def one_hot_to_binary(one_hot_matrix):
    out = np.zeros((one_hot_matrix.shape[0]))
    for i in range(one_hot_matrix.shape[0]):
        out[i] = np.nonzero(one_hot_matrix[i, :])[0]
    return out


def categorical_accuracy_score(y_true, y_pred):
    if np.ndim(y_true) == 2:
        y_true = one_hot_to_binary(y_true)
    if np.ndim(y_pred) == 2:
        y_pred = one_hot_to_binary(y_pred)

    return accuracy_score(y_true, y_pred)


def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]


def spearman_correlation(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def variance_explained_score(y_true, y_pred):
    return np.square(pearson_correlation(y_true, y_pred))


def sensitivity(y_true, y_pred):  # = true positive rate, hit rate, recall
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    else:
        logger.info('Sensitivity (metric) is valid only for binary classification problems. You have ' +
                      str(len(np.unique(y_true))) + ' classes.')
        return np.nan


def specificity(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        logger.info('Specificity (metric) is valid only for binary classification problems. You have ' +
                      str(len(np.unique(y_true))) + ' classes.')
        return np.nan


def balanced_accuracy(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        return (specificity(y_true, y_pred) + sensitivity(y_true, y_pred)) / 2
    else:
        logger.info('Specificity (metric) is valid only for binary classification problems. You have ' +
                      str(len(np.unique(y_true))) + ' classes.')
        return np.nan
