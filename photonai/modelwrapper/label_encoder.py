import numpy as np
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder


class LabelEncoder(SKLabelEncoder):
    """
    Suitable version of the scikit-learn LabelEncoder for PHOTONAI.
    Since the pipeline process streams the underlying samples to
    every transformer, this class is required.

    """
    def __init__(self):
        """Initialize the object."""
        super(LabelEncoder, self).__init__()
        self.needs_y = True

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Call of the underlying sklearn.fit(y) method.

        Parameters:
            X:
                The input samples of shape [n_samples, n_features].

            y:
                The input targets of shape [n_samples, 1].

            **kwargs:
                Ignored input.

        """
        super(LabelEncoder, self).fit(y)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray):
        """
        Call of the underlying sklearn.transform(y) method.

        Parameters:
            X:
                The input samples of shape [n_samples, n_features].

            y:
                The input targets of shape [n_samples, 1].

            **kwargs:
                Ignored input.

        Returns:
            Original X and encoded y.

        """
        yt = super(LabelEncoder, self).transform(y)
        return X, yt

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> (np.ndarray, np.ndarray):
        """
        Call of the underlying sklearn.fit_transform(y) method.

        Parameters:
            X:
                The input samples of shape [n_samples, n_features].

            y:
                The input targets of shape [n_samples, 1].

            **kwargs:
                Ignored input.

        Returns:
            Original X and encoded y.

        """
        return super(LabelEncoder, self).fit_transform(y)
