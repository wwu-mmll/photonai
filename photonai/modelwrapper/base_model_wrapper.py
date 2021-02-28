import numpy as np
from sklearn.base import BaseEstimator


class BaseModelWrapper(BaseEstimator):
    """
    The PHOTONAI interface for implementing custom pipeline elements.

    PHOTONAI works on top of the scikit-learn object API,
    [see documentation](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

    Your class should overwrite the following definitions:

    - `fit(data)`: learn or adjust to the data.

    If it is an estimator, which means it has the ability to learn,

    - it should implement `predict(data)`: using the learned model to generate prediction,
    - should inherit *sklearn.base.BaseEstimator* ([see here](
    http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)),
    - inherits *get_params* and *set_params*.

    If it is an transformer, which means it preprocesses or prepares the data,

    - it should implement `transform(data)`: applying the logic to the data to transform it,
    - should inherit from *sklearn.base.TransformerMixin* ([see here](
    http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)),
    - inherits *fit_transform* as a concatenation of both fit and transform,
    - should inherit *sklearn.base.BaseEstimator* ([see here](
    http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html))
    - inherits *get_params* and *set_params*.

    `Prepare for hyperparameter optimization`

    PHOTONAI expects a `definition for all parameters` you want to optimize in the hyperparameter search in the
    `constructor stub`, and to be addressable with the `same name as class variable`.
    In this way you can define any parameter and it is automatically prepared for the hyperparameter search process.

    See the [scikit-learn object API documentation](
    http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects)
    for more in depth information about the interface.

    """
    def __init__(self):
        pass

    def fit(self, data: np.ndarray, targets: np.ndarray = None):
        """
        Adjust the underlying model or method to the data.

        Parameters:
            data:
                The input samples of shape [n_samples, n_original_features].

            targets:
                The input targets of shape [n_samples, 1].

        Returns:
            IMPORTANT, must return self!

        """

    def predict(self, data: np.ndarray):
        """
        Use the learned model to make predictions.

        Parameters:
            data:
                The input samples of shape [n_samples, n_original_features].

        """

    def transform(self, data: np.ndarray, targets: np.ndarray = None):
        """
        Apply the method's logic to the data.

        Parameters:
            data:
                The input samples of shape [n_samples, n_original_features].

            targets:
                The input targets of shape [n_samples, 1]. Not necessary.

        """

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the models parameters.
        Automatically implemented when inheriting from sklearn.base.BaseEstimator

        Parameters:
            deep:
                If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.
        """
        return super(BaseModelWrapper, self).get_params(deep=deep)

    def set_params(self, **kwargs):
        """
        Takes the given dictionary, with the keys being the variable name,
        and sets the object's parameters to the given values.
        Automatically implemented when inheriting from sklearn.base.BaseEstimator.

        Parameters:
            **kwargs: Estimator parameters.

        """
        super(BaseModelWrapper, self).set_params(**kwargs)
