import unittest
import numpy as np
from photonai.base.PhotonPipeline import PhotonPipeline
from photonai.base.PhotonBase import PipelineElement
from sklearn.datasets import load_breast_cancer


from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
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


class DummyYAndCovariatesTransformer:

    def __init__(self):
        self.needs_y = True
        self.needs_covariates = True

    def fit(self, X, y, **kwargs):
        pass

    def transform(self, X, y, **kwargs):

        self.X = X
        self.y = y
        self.kwargs =kwargs

        y = y + 1
        if len(kwargs) > 0:
            X = X - 1
            kwargs["sample1_edit"] = kwargs["sample1"] + 5
        return X, y, kwargs


class PipelineTests(unittest.TestCase):

    def setUp(self):

        self.X, self.y = load_breast_cancer(True)

        # Photon Version
        self.p_pca = PipelineElement("PCA", {}, random_state=3)
        self.p_svm = PipelineElement("SVC", {}, random_state=3)
        self.p_ss = PipelineElement("StandardScaler", {})

        dummy_element = DummyYAndCovariatesTransformer()
        self.dummy_photon_element = PipelineElement.create("DummyTransformer", dummy_element, {})

        self.sk_pca = PCA(random_state=3)
        self.sk_svc = SVC(random_state=3)
        self.sk_ss = StandardScaler()

    def tearDown(self):
        pass

    def test_regular_use(self):

        photon_pipe = PhotonPipeline([("PCA", self.p_pca), ("SVC", self.p_svm)])
        photon_pipe.fit(self.X, self.y)

        photon_transformed_X, _, _ = photon_pipe.transform(self.X)
        photon_predicted_y = photon_pipe.predict(self.X)

        # the element is given by reference, so it should be fitted right here
        photon_ref_transformed_X, _, _ = self.p_pca.transform(self.X)
        photon_ref_predicted_y = self.p_svm.predict(photon_ref_transformed_X)

        self.assertTrue(np.array_equal(photon_transformed_X, photon_ref_transformed_X))
        self.assertTrue(np.array_equal(photon_predicted_y, photon_ref_predicted_y))

        sk_pipe = SKPipeline([('PCA', self.sk_pca), ("SVC", self.sk_svc)])
        sk_pipe.fit(self.X, self.y)

        sk_predicted_y = sk_pipe.predict(self.X)
        self.assertTrue(np.array_equal(photon_predicted_y, sk_predicted_y))

        # sklearn pipeline does not offer a transform function
        # sk_transformed_X = sk_pipe.transform(X)
        # self.assertTrue(np.array_equal(photon_transformed_X, sk_transformed_X))


    def test_no_estimator(self):

        no_estimator_pipe = PhotonPipeline([("StandardScaler", self.p_ss), ("PCA", self.p_pca)])
        no_estimator_pipe.fit(self.X, self.y)
        photon_no_estimator_transform, _, _ = no_estimator_pipe.transform(self.X)
        photon_no_estimator_predict = no_estimator_pipe.predict(self.X)

        self.assertTrue(np.array_equal(photon_no_estimator_predict, photon_no_estimator_transform))

        self.sk_ss.fit(self.X)
        standardized_data = self.sk_ss.transform(self.X)
        self.sk_pca.fit(standardized_data)
        pca_data = self.sk_pca.transform(standardized_data)

        self.assertTrue(np.array_equal(photon_no_estimator_transform, pca_data))
        self.assertTrue(np.array_equal(photon_no_estimator_predict, pca_data))



    def test_y_and_covariates_transformation(self):

        X = np.ones((200, 50))
        y = np.ones((200,)) + 2
        kwargs = {'sample1': np.ones((200, 5))}

        photon_pipe = PhotonPipeline([("DummyTransformer", self.dummy_photon_element)])

        # if y is none all y transformer should be ignored
        Xt2, yt2, kwargst2 = photon_pipe.transform(X, None, **kwargs)
        self.assertTrue(np.array_equal(Xt2, X))
        self.assertTrue(np.array_equal(yt2, None))
        self.assertTrue(np.array_equal(kwargst2, kwargs))

        # if y is given, all y transformers should be working
        Xt, yt, kwargst = photon_pipe.transform(X, y, **kwargs)

        # assure that data is delivered to element correctly
        self.assertTrue(np.array_equal(X, self.dummy_photon_element.base_element.X))
        self.assertTrue(np.array_equal(y, self.dummy_photon_element.base_element.y))
        self.assertTrue(np.array_equal(kwargs["sample1"], self.dummy_photon_element.base_element.kwargs["sample1"]))

        # assure that data is transformed correctly
        self.assertTrue(np.array_equal(Xt, X - 1))
        self.assertTrue(np.array_equal(yt, y + 1))
        self.assertTrue("sample1_edit" in kwargst)
        self.assertTrue(np.array_equal(kwargst["sample1_edit"], kwargs["sample1"] + 5))


    def test_predict_with_training_flag(self):
        # manually edit labels
        sk_pipe = SKPipeline([("SS", self.sk_ss), ("SVC", self.sk_svc)])
        y_plus_one = self.y + 1
        sk_pipe.fit(self.X, y_plus_one)
        sk_pred = sk_pipe.predict(self.X)

        # edit labels during pipeline
        p_pipe = PhotonPipeline([("SS", self.p_ss), ("YT", self.dummy_photon_element), ("SVC", self.p_svm)])
        p_pipe.fit(self.X, self.y)
        p_pred = p_pipe.predict(self.X)

        sk_standardized_X = self.sk_ss.transform(self.X)
        input_of_y_transformer = self.dummy_photon_element.base_element.X
        self.assertTrue(np.array_equal(sk_standardized_X, input_of_y_transformer))

        self.assertTrue(np.array_equal(sk_pred, p_pred))
