import unittest
from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np


class ConfounderRemovalTests(unittest.TestCase):

    def setUp(self):

        self.X, self.y = load_breast_cancer(True)
        self.pipe = Hyperpipe("confounder_pipe", inner_cv=KFold(n_splits=3),
                              metrics=["accuracy"], best_config_metric="accuracy")
        self.pipe += PipelineElement("StandardScaler")
        self.cr = PipelineElement("ConfounderRemoval")
        self.pipe += self.cr
        self.pipe += PipelineElement("SVC")
        self.random_confounders = np.random.randn(self.X.shape[0], 1)

    def tearDown(self):
        pass

    def test_use(self):
        self.pipe.fit(self.X, self.y, **{'covariates': self.random_confounders})
        trans_data = self.pipe.transform(self.X, **{'covariates': self.random_confounders})

    def test_dimensions(self):
        with self.assertRaises(ValueError):
            self.cr.fit(self.X, self.y, covariates=np.random.randn(self.X.shape[0]-10, 2))

    def test_key_error(self):

        with self.assertRaises(KeyError):
