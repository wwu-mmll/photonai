from sklearn.base import BaseEstimator, ClassifierMixin


class WrapperTestElement(BaseEstimator, ClassifierMixin):
    """
    This ugly class is used to test the data streaming through the pipeline
    """

    def __init__(self, any_param=1):
        self.data_dict = {}
        self.any_param = any_param

    def fit(self, X, y):
        self.data_dict['fit_X'] = X
        self.data_dict['fit_y'] = y
        return self

    def transform(self, X):
        self.data_dict['transform_X'] = X
        return X

    def predict(self, X):
        self.data_dict['predict_X'] = X
        return X

    def fit_transform(self, X, y):
        self.data_dict['fit_transform_X'] = X
        self.data_dict['fit_transform_y'] = y
        self.fit(X, y)
        return self.transform(X)

    def fit_predict(self, X, y):
        self.data_dict['fit_predict_X'] = X
        self.data_dict['fit_predict_y'] = y
        self.fit(X, y)
        return self.predict(X)
