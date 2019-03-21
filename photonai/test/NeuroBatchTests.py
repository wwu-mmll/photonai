import unittest

class DummyBatchTransformer:

    def __init__(self):
        self.needs_y = False
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        pass

    def transform(self, X, y, **kwargs):

        X_new = [str(x) + str(y[i]) for i, x in enumerate(X)]
        if len(kwargs) > 0:
            X = X - 1
            kwargs["sample1_edit"] = kwargs["sample1"] + 5
        return X, y, kwargs


class PipelineTests(unittest.TestCase):
    pass
