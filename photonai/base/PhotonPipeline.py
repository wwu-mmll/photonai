from sklearn.utils.metaestimators import _BaseComposition
import numpy as np


class PhotonPipeline(_BaseComposition):

    def __init__(self, steps):
        self.steps = steps

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('steps', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params('steps', **kwargs)

        return self

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None:
                continue
            if not (hasattr(t, "fit") or not hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if estimator is not None and not hasattr(estimator, "fit"):
            raise TypeError("Last step of Pipeline should implement fit. "
                            "'%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    def fit(self, X, y=None, **kwargs):

        self._validate_steps()

        for (name, transformer) in self.steps[:-1]:
            transformer.fit(X, y, **kwargs)
            X, y, kwargs = transformer.transform(X, y, **kwargs)

        if self._final_estimator is not None:
            self._final_estimator.fit(X, y, **kwargs)

        return self

    def check_for_numpy_array(self, list_object):
        # be compatible to list of (image-) files
        if isinstance(list_object, list):
            return np.asarray(list_object)
        else:
            return list_object

    def transform(self, X, y=None, **kwargs):
        """
        Calls transform on every step that offers a transform function
        including the last step if it has the transformer flag,
        and excluding the last step if it has the estimator flag but no transformer flag.

        Returns transformed X, y and kwargs
        """
        for (name, transformer) in self.steps[:-1]:
            X, y, kwargs = transformer.transform(X, y, **kwargs)

        if self._final_estimator is not None:
            if self._final_estimator.is_transformer and not self._final_estimator.is_estimator:
                X, y, kwargs = self._final_estimator.transform(X, y, **kwargs)
                # always work with numpy arrays to avoid checking for shape attribute
                X = self.check_for_numpy_array(X)
                y = self.check_for_numpy_array(y)
        return X, y, kwargs

    def predict(self, X, training=False, **kwargs):
        """
        Transforms the data for every step that offers a transform function
        and then calls the estimator with predict on transformed data.
        It returns the predictions made.

        In case the last step is no estimator, it returns the transformed data.
        """
        # first transform
        if not training:
            X, _, kwargs = self.transform(X, y=None, **kwargs)

        # then call predict on final estimator
        if self._final_estimator is not None:
            if self._final_estimator.is_estimator:
                y_pred = self._final_estimator.predict(X, **kwargs)
                return y_pred
            else:
                return X
        else:
            return None

    def predict_proba(self, X, training: bool=False, **kwargs):
        if not training:
            X, _, kwargs = self.transform(X, y=None, **kwargs)

        if self._final_estimator is not None:
            if self._final_estimator.is_estimator:
                if hasattr(self._final_estimator, "predict_proba"):
                    if hasattr(self._final_estimator, 'needs_covariates'):
                        if self._final_estimator.needs_covariates:
                            return self._final_estimator.predict_proba(X, **kwargs)
                        else:
                            return self._final_estimator.predict_proba(X)
                    else:
                        return self._final_estimator.predict_proba(X)

        raise NotImplementedError("The final estimator does not have a predict_proba method")

    def inverse_transform(self, X, y=None, **kwargs):
        # simply use X to apply inverse_transform
        # does not work on any transformers changing y or kwargs!
        for name, transform in self.steps[::-1]:
            if hasattr(transform, 'inverse_transform'):
                X, y, kwargs = transform.inverse_transform(X, y, **kwargs)
        return X, y, kwargs

    def fit_transform(self, X, y=None, **kwargs):
        # return self.fit(X, y, **kwargs).transform(X, y, **kwargs)
        raise NotImplementedError('fit_transform not yet implemented in PHOTON Pipeline')

    def fit_predict(self, X, y=None, **kwargs):
        raise NotImplementedError('fit_predict not yet implemented in PHOTON Pipeline')


    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]


