from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder


class LabelEncoder(SKLabelEncoder, BaseEstimator, TransformerMixin):

    def __init__(self):
        super(LabelEncoder, self).__init__()
        self.needs_y = True

    def fit(self, X, y=None, **kwargs):
        super(LabelEncoder, self).fit(y)
        return self

    def transform(self, X, y=None, **kwargs):
        yt = super(LabelEncoder, self).transform(y)
        return X, yt

    def fit_transform(self, X, y=None, **kwargs):
        return super(LabelEncoder, self).fit_transform(y)
