import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from photonai.modelwrapper.PhotonOneClassSVM import PhotonOneClassSVM
from photonai.modelwrapper.RangeRestrictor import RangeRestrictor
from photonai.modelwrapper.PhotonMLPClassifier import MLPClassifier


class ModelWrapperTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)

    def test_photon_mlp(self):
        mlp = MLPClassifier()
        mlp.fit(self.X, self.y)
        mlp.predict(self.X)
        mlp.predict_proba(self.X)

    def test_photon_one_class_svm(self):
        osvm = PhotonOneClassSVM()
        osvm.fit(self.X, self.y)
        osvm.predict(self.X)

    def test_range_restrictor(self):
        rr = RangeRestrictor()
        rr.fit(self.X, self.y)
        pred = rr.predict(self.X)
        self.assertEqual(np.min(pred), 0)
        self.assertEqual(np.max(pred), 100)
