from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder


class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.label_encoder_object = SKLabelEncoder()
        self.needs_y = True

    def fit(self, X, y=None, **kwargs):
        self.label_encoder_object.fit(y)
        return self

    def transform(self, X, y=None, **kwargs):
        yt = self.label_encoder_object.transform(y)
        return X, yt
