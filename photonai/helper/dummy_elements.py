from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class DummyYAndCovariatesTransformer(BaseEstimator):

    def __init__(self):
        self.needs_y = True
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        pass

    def transform(self, X, y, **kwargs):

        self.X = X
        self.y = y
        self.kwargs =kwargs

        if y is not None:
            y = y + 1
        if len(kwargs) > 0:
            X = X - 1
            kwargs["sample1_edit"] = kwargs["sample1"] + 5
        return X, y, kwargs


class DummyEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return X + 1


class DummyNeedsCovariatesEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return X + 1


class DummyTransformer(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X + 1


class DummyNeedsCovariatesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X + 1, {'covariates': kwargs['covariates'] + 1}


class DummyNeedsYTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.needs_y = True

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X, y):
        return X + 1, y + 1


class DummyNeedsCovariatesAndYTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.needs_y = True
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X, y, **kwargs):
        return X + 1, y + 1, {'covariates': kwargs['covariates'] + 1}


class DummyEstimatorWrongType(BaseEstimator):
    _estimator_type = 'clusterer'

    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return X


class DummyTransformerWithPredict(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return X


class DummyEstimatorNoPredict(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        return self


