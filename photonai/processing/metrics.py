"""
Define custom metrics here
The method stub of all metrics is
function_name(y_true, y_pred)
"""
from typing import List, Callable, Union, Dict  # Hashable
import numpy as np
from scipy.stats import spearmanr
from photonai.photonlogger.logger import logger

from sklearn.metrics import accuracy_score


class Scorer(object):
    """
    Transforms a string literal into an callable instance of a particular metric
    BHC 0.1 - support cluster scoring by add clustering scores and type
            - added ELEMENT_TYPES
            - added SCORE_TYPES
            note: really ELEMENT_SCORE
    """

    ELEMENT_TYPES = ["Classification", "Regression", "Clustering"]
    CLSFID = 0
    REGRID = 1
    CLSTID = 2

    METRIC_PKGID = 0
    METRIC_NAMEID = 1
    METRIC_SCORE_TYPEID = 2
    METRIC_SIGNID = 3

    SCORE_TYPES = ["score", "error", "unsupervised"]
    SCOREID = 0
    ERRORID = 1
    UNSUPERID = 2

    SCORE_SIGN = [1, 0, -1, np.nan]
    SCORE_POSID = 0  # FOR OPTIMIXATION , GREATER IS BETTER
    SCORE_ZEROID = 1  # NO PREDICTION. no y_pred ,
    SCORE_NEGID = 2  # FOR OPTIMIXATION , GREATER IS BETTER
    SCORE_NF = 3  # scores (metrics( not found

    ELEMENT_SCORES = {
        ELEMENT_TYPES[CLSFID]: {  # Classification
            "matthews_corrcoef": (
                "sklearn.metrics",
                "matthews_corrcoef",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "accuracy": (
                "sklearn.metrics",
                "accuracy_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "f1_score": (
                "sklearn.metrics",
                "f1_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "hamming_loss": (
                "sklearn.metrics",
                "hamming_loss",
                SCORE_TYPES[ERRORID],
                SCORE_NEGID,
            ),
            "log_loss": (
                "sklearn.metrics",
                "log_loss",
                SCORE_TYPES[ERRORID],
                SCORE_NEGID,
            ),
            "precision": (
                "sklearn.metrics",
                "precision_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "recall": (
                "sklearn.metrics",
                "recall_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "auc": (
                "sklearn.metrics",
                "roc_auc_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "sensitivity": (
                "photonai.processing.metrics",
                "sensitivity",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "specificity": (
                "photonai.processing.metrics",
                "specificity",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "balanced_accuracy": (
                "photonai.processing.metrics",
                "balanced_accuracy",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "categorical_accuracy": (
                "photonai.processing.metrics",
                "categorical_accuracy_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
        },
        ELEMENT_TYPES[REGRID]: {  # Regression
            "mean_squared_error": (
                "sklearn.metrics",
                "mean_squared_error",
                SCORE_TYPES[ERRORID],
                SCORE_NEGID,
            ),
            "mean_absolute_error": (
                "sklearn.metrics",
                "mean_absolute_error",
                SCORE_TYPES[ERRORID],
                SCORE_NEGID,
            ),
            "explained_variance": (
                "sklearn.metrics",
                "explained_variance_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "r2": ("sklearn.metrics", "r2_score", SCORE_TYPES[SCOREID], SCORE_POSID),
            "pearson_correlation": (
                "photonai.processing.metrics",
                "pearson_correlation",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "spearman_correlation": (
                "photonai.processing.metrics",
                "spearman_correlation",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            "variance_explained": (
                "photonai.processing.metrics",
                "variance_explained_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
        },
        ELEMENT_TYPES[CLSTID]: {  # Clustering
            ### supervised clustering metrics from sklearn.metrics
            # ['ARI'] = metrics.adjusted_rand_score(y, labels)
            "ARI": (
                "sklearn.metrics",
                "adjusted_rand_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            # ['MI'] = metrics.adjusted_mutual_info_score(y, labels)
            "MI": (
                "sklearn.metrics",
                "adjusted_mutual_info_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            # ['HCV'] = metrics.homogeneity_score(y, labels)
            "HCV": (
                "sklearn.metrics",
                "homogeneity_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            # ['FM'] = metrics.fowlkes_mallows_score(y, labels)
            "FM": (
                "sklearn.metrics",
                "fowlkes_mallows_score",
                SCORE_TYPES[SCOREID],
                SCORE_POSID,
            ),
            ### un-supervised clustering metrics from sklearn.metrics
            # ['SC'] = metrics.silhouette_score(X, labels, metric='euclidean')
            "SC": (
                "sklearn.metrics",
                "silhouette_score",
                SCORE_TYPES[UNSUPERID],
                SCORE_ZEROID,
            ),
            # ['CH'] = metrics.calinski_harabaz_score(X, labels)
            "CH": (
                "sklearn.metrics",
                "calinski_harabaz_score",
                SCORE_TYPES[UNSUPERID],
                SCORE_ZEROID,
            ),
            # ['DB'] = metrics.davies_bouldin_score(X, labels)
            "DB": (
                "sklearn.metrics",
                "davies_bouldin_score",
                SCORE_TYPES[UNSUPERID],
                SCORE_ZEROID,
            ),
        },
    }

    @staticmethod
    def flatten(d):
        v = [[i] if not isinstance(i, list) else flatten(i) for i in d]
        return [i for b in v for i in b]

    METRICS = {}
    for m in ELEMENT_SCORES.keys():
        METRICS.update(ELEMENT_SCORES[m])

    # 1.1
    @staticmethod
    def is_metric(metric: str) -> bool:
        """
        :raises
        if not known metric

        :param metric:
        :return: True
        """
        if metric in Scorer.METRICS:
            return True
        else:
            logger.error(
                "Specify valid metric to choose best config: {}".format(metric)
            )
        raise NameError(
            "Specify valid metric to choose best config:  {}".format(metric)
        )

    # 1.1
    @classmethod
    def create(cls, metric: str) -> Callable:
        """
        Searches for the metric by name and instantiates the according calculation function
        :param metric: the name of the metric as encoded in the METRICS
        :type metric: str
        :return: a callable instance of the metric calculation
        """
        # create -> error  if not known error
        if Scorer.is_metric(metric):
            imported_module = __import__(
                Scorer.METRICS[metric][Scorer.METRIC_PKGID],
                globals(),
                locals(),
                Scorer.METRICS[metric][Scorer.METRIC_NAMEID],
                0,
            )
            return getattr(
                imported_module, Scorer.METRICS[metric][Scorer.METRIC_NAMEID]
            )

    @staticmethod
    def metric_sign(metric: str) -> int:  # <<<-- greater_is_better_distinction
        """
        Only look in Scorer.METRICS (a list of metrics) and
        return  scorer sign.  +1, 0, -1
        :param metric:
        :return:
        """
        Scorer.is_metric(metric)
        return Scorer.SCORE_SIGN[Scorer.METRICS[metric][Scorer.METRIC_SIGNID]]

    @staticmethod
    def calculate_metrics(
        y_true: Union[List[float], List[List[float]]],
        y_pred: List[int],
        metrics: List[str],
    ) -> Dict[str, float]:
        """
        classification, regression  y_true, y_pred
        ------------------------------------------
        Applies all metrics to the given predicted and true values.
        (or 2-d array, clustrer assignment labels
        The metrics are encoded via a string literal which is mapped
        to the according calculation function,

        :param y_true: the truth values
        :param y_pred: the predicted values
        :param metrics:
        :return: dict of metrics values

        clustering  X, labels
        ----------------------
        :param y_true: X array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a single data point.

        :param y_true: labels array-like, shape (n_samples,)
        Predic1ted labels for each sample.
        """
        output_metrics = {}
        if not metrics:
            return output_metrics

        for metric in metrics:
            # create -> error  if not known error, then calc score value
            output_metrics[metric] = Scorer.create(metric)(y_true, y_pred)

        return output_metrics


def binary_to_one_hot(binary_vector):
    classes = np.unique(binary_vector)
    out = np.zeros((binary_vector.shape[0], len(classes)), dtype=np.int)
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
        logger.info(
            "Sensitivity (metric) is valid only for binary classification problems. You have "
            + str(len(np.unique(y_true)))
            + " classes."
        )
        return np.nan


def specificity(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        logger.info(
            "Specificity (metric) is valid only for binary classification problems. You have "
            + str(len(np.unique(y_true)))
            + " classes."
        )
        return np.nan


def balanced_accuracy(y_true, y_pred):  # = true negative rate
    if len(np.unique(y_true)) == 2:
        return (specificity(y_true, y_pred) + sensitivity(y_true, y_pred)) / 2
    else:
        logger.info(
            "Specificity (metric) is valid only for binary classification problems. You have "
            + str(len(np.unique(y_true)))
            + " classes."
        )
        return np.nan
