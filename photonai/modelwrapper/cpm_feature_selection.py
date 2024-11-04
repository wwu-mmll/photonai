import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import beta, spearmanr

from photonai.photonlogger.logger import logger


class CPMFeatureSelection(BaseEstimator, TransformerMixin):
    """Feature Selection using Connectome-Based Predictive Modeling.
    loosely based on this paper https://www.nature.com/articles/nprot.2016.178#Sec10

    Correlate all features with target and select significant features only.
    Sum significant edges for positive correlations and negative correlations separately.
    """
    _estimator_type = "transformer"

    def __init__(self, p_threshold: float = .05, corr_method: str = 'pearson'):
        """
        Initialize the object.

        Parameters:
            p_threshold:
                Upper bound for p_values.
            corr_method:
                Correlation coefficient method. Can be 'pearson' or 'spearman'.

        """
        self.p_threshold = p_threshold
        self.corr_method = corr_method
        if corr_method not in ['pearson', 'spearman']:
            raise NotImplementedError("corr_method has to be either 'pearson' or 'spearman'.")

        self.significant_edges = None
        self.positive_edges = None
        self.negative_edges = None
        self.n_original_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Calculate correlation coefficients between features of X and y.

        Parameters:
            X:
                The input samples of shape [n_samples, n_original_features]

            y:
                The input targets of shape [n_samples, 1]

        """
        n_samples, self.n_original_features = X.shape

        if self.corr_method == 'pearson':
            corr = self._columnwise_pearson
        elif self.corr_method == 'spearman':
            corr = self._columnwise_spearman
        else:
            corr = None

        r, p = corr(X, y)
        self.significant_edges = p < self.p_threshold
        self.positive_edges = r > 0
        self.negative_edges = r < 0
        return self

    @staticmethod
    def _columnwise_pearson(X, y):
        """
        Compute Pearson's correlation coefficient between y and every column of X efficiently

        :param X: ndarray
        :param y: ndarray
        :return: r_values: array of correlation coefficients
                 p_values: array of corresponding p-values
        """
        n_samples = X.shape[0]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        r_values = np.dot(X.T, y) / n_samples

        # I used the p-value calculation described here
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        dist = beta(n_samples / 2 - 1, n_samples / 2 - 1, loc=-1, scale=2)
        p_values = 2 * dist.cdf(-np.abs(r_values))
        return r_values, p_values

    @staticmethod
    def _columnwise_spearman(X, y):
        # ToDo: make more efficient by not relying on for loop
        n_features = X.shape[1]
        r_values, p_values = np.zeros(n_features), np.zeros(n_features)
        for i in range(n_features):
            corr = spearmanr(X[:, i], y)
            r_values[i], p_values[i] = corr.statistic, corr.pvalue
        return r_values, p_values

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Sum over significant positive and significant negative edges.

        Parameters:
            X
                The input samples of shape [n_samples, n_original_features]

        Returns:
            array of shape [n_samples, 2].

        """
        return np.stack([np.sum(X[:, (self.significant_edges == self.positive_edges)], axis=1),
                         np.sum(X[:, (self.significant_edges == self.negative_edges)], axis=1)], axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse to original dimension.

        Parameters:
            X:
                The input samples of shape [n_samples, 2].

        Returns:
            Array of shape [1, n_original_features]
            with columns of zeros inserted where features haven't been included in the sum of positive or
            negative edges. First value of input is inserted where a significant positive edge had been identified.
            Second value of the input is inserted where a significant negative edge had been identified.

        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 2:
            msg = "X needs to have 2 features (which correspond to the sum of positive and negative edges)."
            logger.error(msg)
            raise ValueError(msg)

        if X.shape[0] > 1:
            msg = "X can only contain one array with shape [1, 2]."
            logger.error(msg)
            raise ValueError(msg)

        Xt = np.zeros((X.shape[0], self.n_original_features))
        Xt[:, (self.significant_edges == self.positive_edges)] = X[:, 0]
        Xt[:, (self.significant_edges == self.negative_edges)] = X[:, 1]
        return Xt
