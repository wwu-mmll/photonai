from sklearn.base import BaseEstimator, ClassifierMixin


class WrapperModel(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=1, model=None):
        self.learning_rate = learning_rate
        self.model = model

    def fit(self, data, targets):
        # whatever needs to be done ....
        # do anything with self.learning_rate
        # self.model.fit(data, targets)
        print('Hello from Wrapper Model. fitting with current learning_rate:', self.learning_rate)
        return self

    @staticmethod
    def transform(data):
        return data

    # def predict(self, data):
    #     return self.model.predict(data)

    def set_params(self, **params):
        print('setting params:', str(params))
        super(WrapperModel, self).set_params(**params)






