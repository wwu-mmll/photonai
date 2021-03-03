"""
Define custom metrics here
The method stub of all metrics is
function_name(y_true, y_pred)
"""
import warnings
import numpy as np
from typing import Union, Type, Callable, Optional, Tuple, Dict
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

from photonai.photonlogger.logger import logger


class Scorer(object):
    """Scorer.

    Transforms a string literal into an callable instance of a particular metric.

    """

    ELEMENT_DICTIONARY: Dict[str, Tuple[str, str, str]] = {
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

    CUSTOM_ELEMENT_DICTIONARY: Dict[str, Callable] = {}

    Metric_Type = Union[
        Callable,
        'keras.metrics.Metric',
        Type['keras.metrics.Metric']
    ]

    dynamic_keras_import = None

    @classmethod
    def try_import_keras(cls):
        try:
            cls.dynamic_keras_import = __import__('keras')
        except ImportError:
            pass

    @classmethod
    def register_custom_metric(cls, metric: Union[Metric_Type, Tuple[str, Metric_Type]]) -> Optional[str]:
        if cls.dynamic_keras_import is None:
            cls.try_import_keras()
        # if metric is already a string, don't do anything
        if isinstance(metric, str):
            return metric

        # derive name from metric class unless it is explicitly given with a tuple
        if metric is not None:
            if isinstance(metric, Tuple):
                metric_name = metric[0]
                metric = metric[1]
            elif cls.dynamic_keras_import is not None and isinstance(metric, cls.dynamic_keras_import.metrics.Metric):
                metric_name = "custom_" + str(metric.__module__) + '.' + str(type(metric).__name__)
            else:
                metric_name = "custom_" + str(metric.__module__) + '.' + str(metric.__name__)
            metric_name = metric_name.lower()
        else:
            raise ValueError("Metric is None")

        # Check if metric_name is already registered
        if metric_name in Scorer.CUSTOM_ELEMENT_DICTIONARY:
            warn_text = 'Custom metric name ' + metric_name + ' is ambiguous. Please specify metric as tuple with ' + \
                        'cooresponding name (e.g. instead of metrics=[keras.metrics.Accuracy] use ' \
                        'metrics=[(\'MetricName1\', keras.metrics.Accuracy)]. Only the first occurance of this ' \
                        'metric will be used!'
            logger.warning(warn_text)
            warnings.warn(warn_text)
            return None

        # derive a metric function from the given object
        if cls.dynamic_keras_import is not None and (
                (isinstance(metric, type) and issubclass(metric, cls.dynamic_keras_import.metrics.Metric))
                or isinstance(metric, cls.dynamic_keras_import.metrics.Metric)
        ):
            if isinstance(metric, type) and issubclass(metric, cls.dynamic_keras_import.metrics.Metric):
                metric_obj = metric()
            else:
                metric_obj = metric

            def metric_func(y_true, y_pred):
                metric_obj.reset_states()
                metric_obj.update_state(y_true=y_true, y_pred=y_pred)
                return float(cls.dynamic_keras_import.backend.eval(metric_obj.result()))

            Scorer.CUSTOM_ELEMENT_DICTIONARY[metric_name] = metric_func
        elif callable(metric):
            Scorer.CUSTOM_ELEMENT_DICTIONARY[metric_name] = metric
        return metric_name
    
    @classmethod
    def create(cls, metric: str) -> Optional[Callable]:
        """Searches for the metric by name and instantiates the according calculation function

        Parameters
        ----------
        metric: str
            The name of the metric as encoded in the ELEMENT_DICTIONARY.

        Returns
        -------
        metric_function: Callable
            A callable instance of the metric calculation.

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
                msg = 'Could not find according class: ' + Scorer.ELEMENT_DICTIONARY[metric]
                logger.error(msg)
                raise AttributeError(msg)
        elif metric in Scorer.CUSTOM_ELEMENT_DICTIONARY:
            return Scorer.CUSTOM_ELEMENT_DICTIONARY[metric]
        else:
            msg = 'Metric not supported right now:' + str(metric)
            logger.error(msg)
            raise NameError(msg)

    @staticmethod
    def greater_is_better_distinction(metric: str) -> bool:
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
        elif metric in Scorer.CUSTOM_ELEMENT_DICTIONARY:
            # Check if it's a greater_is_better metric by calling it with example values
            metric_callable = Scorer.create(metric)
            y_true = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
            y_pred = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
            true_true = metric_callable(y_true, y_true)
            true_pred = metric_callable(y_true, y_pred)
            return true_true > true_pred
        else:
            logger.error('Specify valid metric to choose best config.')
        raise NameError('Specify valid metric to choose best config.')

    @staticmethod
    def calculate_metrics(y_true, y_pred, metrics):
        """Applies all metrics to the given predicted and true values.
        The metrics are encoded via a string literal which is mapped
        to the according calculation function.

        Parameters
        ----------
        y_true: list
            The truth values.
        y_pred: list
            The predicted values.
        metrics: list of str
            List of all metrics to be calculated from y_true and y_pred.

        Returns
        --------
        result: dict
            Dictionary with format name_of_metric -> value.

        """

        # Todo: HOW TO CHECK IF ITS REGRESSION?!
        # The following works only for classification
        # if np.ndim(y_pred) == 2:
        #     y_pred = one_hot_to_binary(y_pred)
        #     logger.warning("test_predictions was one hot encoded => transformed to binary")
        #
        # if np.ndim(y_true) == 2:
        #     y_true = one_hot_to_binary(y_true)
        #     logger.warning("test_y was one hot encoded => transformed to binary")

        output_metrics = {}
        if metrics:
            for metric in metrics:
                scorer = Scorer.create(metric)
                if scorer is not None:
                    scorer_value = scorer(y_true, y_pred)
                    output_metrics[metric] = scorer_value
                else:
                    output_metrics[metric] = np.nan

        return output_metrics


def one_hot_to_binary(one_hot_matrix):
    out = np.zeros((one_hot_matrix.shape[0]))
    for i in range(one_hot_matrix.shape[0]):
        out[i] = np.nonzero(one_hot_matrix[i, :])[0][0]
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
