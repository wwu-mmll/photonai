from sklearn.base import BaseEstimator, ClassifierMixin


class WrapperModel(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.model = model

    def fit(self, data, targets):
        # whatever needs to be done ....
        # do anything with self.learning_rate
        self.model.fit(data, targets)
        return self

    def predict(self, data):
        return self.model.predict(data)





