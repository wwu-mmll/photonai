
class BaseModelWrapper:
    """
    The PHOTON interface for implementing custom pipeline elements.

    PHOTON works on top of the scikit-learn object API,
    [see documentation](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects)

    Your class should overwrite the following definitions:

    - `fit(data)`: learn or adjust to the data

    If it is an estimator, which means it has the ability to learn,

    - it should implement `predict(data)`: using the learned model to generate prediction
    - should inherit *sklearn.base.BaseEstimator* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html))
    - inherits *get_params* and *set_params*

    If it is an transformer, which means it preprocesses or prepares the data

    - it should implement `transform(data)`: applying the logic to the data to transform it
    - should inherit from *sklearn.base.TransformerMixin* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html))
    - inherits *fit_transform* as a concatenation of both fit and transform
    - should inherit *sklearn.base.BaseEstimator* ([see here](http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html))
    - inherits *get_params* and *set_params*

    `Prepare for hyperparameter optimization`

    PHOTON expects you to `define all parameters` that you want to optimize in the hyperparameter search in the
    `constructor stub`, and to be addressable with the `same name as class variable`.
    In this way you can define any parameter and it is automatically prepared for the hyperparameter search process.

    See the [scikit-learn object API documentation](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) for more in depth information about the interface.
    """

    def __init__(self):
        pass

    def fit(self, data, targets=None):
        """
        Adjust the underlying model or method to the data.

        Returns
        -------
        IMPORTANT: must return self!
        """

    def predict(self, data):
        """
        Use the learned model to make predictions.
        """

    def transform(self, data):
        """
        Apply the method's logic to the data.
        """

    def get_params(self, deep=True):
        """
        Get the models parameters.
        Automatically implemented when inheriting from sklearn.base.BaseEstimator
        """

    def set_params(self, *kwargs):
        """
        Takes the given dictionary, with the keys being the variable name, and sets the object's parameters to the given values.
        Automatically implemented when inheriting from sklearn.base.BaseEstimator
        """
