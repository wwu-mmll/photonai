import unittest
import numpy as np
from photonai.base.PhotonPipeline import PhotonPipeline
from photonai.base.PhotonBase import PipelineElement
from sklearn.datasets import load_breast_cancer


# assertEqual(a, b) 	a == b
# assertNotEqual(a, b) 	a != b
# assertTrue(x) 	bool(x) is True
# assertFalse(x) 	bool(x) is False
# assertIs(a, b) 	a is b 	3.1
# assertIsNot(a, b) 	a is not b 	3.1
# assertIsNone(x) 	x is None 	3.1
# assertIsNotNone(x) 	x is not None 	3.1
# assertIn(a, b) 	a in b 	3.1
# assertNotIn(a, b) 	a not in b 	3.1
# assertIsInstance(a, b) 	isinstance(a, b) 	3.2
# assertNotIsInstance(a, b) 	not isinstance(a, b) 	3.2


class PipelineTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_regular_use(self):

        X, y = load_breast_cancer(True)

        # Photon Version
        p_pca = PipelineElement("PCA", {}, random_state=3)
        p_svm = PipelineElement("SVC", {}, random_state=3)

        photon_pipe = PhotonPipeline([("PCA", p_pca), ("SVC", p_svm)])
        photon_pipe.fit(X, y)

        photon_transformed_X = photon_pipe.transform(X)
        photon_predicted_y = photon_pipe.predict(X)

        # the element is given by reference, so it should be fitted right here
        photon_ref_transformed_X = p_pca.transform(X)
        photon_ref_predicted_y = p_svm.predict(photon_ref_transformed_X)

        self.assertTrue(np.array_equal(photon_transformed_X, photon_ref_transformed_X))
        self.assertTrue(np.array_equal(photon_predicted_y, photon_ref_predicted_y))

        from sklearn.decomposition.pca import PCA
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline as SKPipeline
        sk_pca = PCA(random_state=3)
        sk_svc = SVC(random_state=3)

        sk_pipe = SKPipeline([('PCA', sk_pca), ("SVC", sk_svc)])
        sk_pipe.fit(X, y)

        sk_predicted_y = sk_pipe.predict(X)
        self.assertTrue(np.array_equal(photon_predicted_y, sk_predicted_y))

        # sklearn pipeline does not offer a transform function
        # sk_transformed_X = sk_pipe.transform(X)
        # self.assertTrue(np.array_equal(photon_transformed_X, sk_transformed_X))



