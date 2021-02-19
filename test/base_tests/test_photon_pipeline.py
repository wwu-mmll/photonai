import glob
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from photonai.base import PipelineElement, Preprocessing, Stack, Switch, Branch, \
    CallbackElement
from photonai.base.cache_manager import CacheManager
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.helper.dummy_elements import DummyYAndCovariatesTransformer
from photonai.helper.photon_base_test import PhotonBaseTest


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


class PipelineTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(PipelineTests, cls).setUpClass()

    def setUp(self):
        super(PipelineTests, self).setUp()
        self.X, self.y = load_breast_cancer(return_X_y=True)

        # Photon Version
        self.p_pca = PipelineElement("PCA", {}, random_state=3)
        self.p_svm = PipelineElement("SVC", {}, random_state=3)
        self.p_ss = PipelineElement("StandardScaler", {})
        self.p_dt = PipelineElement("DecisionTreeClassifier", random_state=3)

        dummy_element = DummyYAndCovariatesTransformer()
        self.dummy_photon_element = PipelineElement.create("DummyTransformer", dummy_element, {})

        self.sk_pca = PCA(random_state=3)
        self.sk_svc = SVC(random_state=3)
        self.sk_ss = StandardScaler()
        self.sk_dt = DecisionTreeClassifier(random_state=3)

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

    def test_add_preprocessing(self):
        my_preprocessing = Preprocessing()
        my_preprocessing += PipelineElement('LabelEncoder')
        photon_pipe = PhotonPipeline([("PCA", self.p_pca), ("SVC", self.p_svm)])
        photon_pipe.add_preprocessing(my_preprocessing)

        self.assertEqual(len(photon_pipe.named_steps), 3)
        first_element = photon_pipe.elements[0][1]
        self.assertTrue(first_element == my_preprocessing)
        self.assertTrue(photon_pipe.named_steps['Preprocessing'] == my_preprocessing)

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

    def test_inverse_tansform(self):
        # simple pipe
        sk_pipe = SKPipeline([("SS", self.sk_ss), ("PCA", self.sk_pca)])
        sk_pipe.fit(self.X, self.y)
        sk_transform = sk_pipe.transform(self.X)
        sk_inverse_transformed = sk_pipe.inverse_transform(sk_transform)

        photon_pipe = PhotonPipeline([("SS", self.p_ss), ("PCA", self.p_pca)])
        photon_pipe.fit(self.X, self.y)
        p_transform, _, _ = photon_pipe.transform(self.X)
        p_inverse_transformed, _, _ = photon_pipe.inverse_transform(p_transform)

        self.assertTrue(np.array_equal(sk_inverse_transformed, p_inverse_transformed))

        # now including stack
        stack = Stack('stack', [self.p_pca])
        stack_pipeline = PhotonPipeline([("stack", stack), ('StandardScaler', PipelineElement('StandardScaler')),
                                         ('LinearSVC', PipelineElement('LinearSVC'))])
        stack_pipeline.fit(self.X, self.y)
        feature_importances = stack_pipeline.feature_importances_
        inversed_data, _, _ = stack_pipeline.inverse_transform(feature_importances)
        self.assertEqual(inversed_data.shape[1], self.X.shape[1])

    # Todo: add tests for kwargs

    def test_predict_proba(self):

        sk_pipe = SKPipeline([("SS", self.sk_ss), ("SVC", self.sk_dt)])
        sk_pipe.fit(self.X, self.y)
        sk_proba = sk_pipe.predict_proba(self.X)

        photon_pipe = PhotonPipeline([("SS", self.p_ss), ("SVC", self.p_dt)])
        photon_pipe.fit(self.X, self.y)
        photon_proba = photon_pipe.predict_proba(self.X)

        self.assertTrue(np.array_equal(sk_proba, photon_proba))

    def test_copy_me(self):
        switch = Switch("my_copy_switch")
        switch += PipelineElement("StandardScaler")
        switch += PipelineElement("RobustScaler", test_disabled=True)

        stack = Stack("RandomStack")
        stack += PipelineElement("SVC")
        branch = Branch('Random_Branch')
        pca_hyperparameters = {'n_components': [5, 10]}
        branch += PipelineElement("PCA", hyperparameters=pca_hyperparameters)
        branch += PipelineElement("DecisionTreeClassifier")
        stack += branch

        photon_pipe = PhotonPipeline([("SimpleImputer", PipelineElement("SimpleImputer")),
                                      ("my_copy_switch", switch),
                                      ('RandomStack', stack),
                                      ('Callback1', CallbackElement('tmp_callback', np.mean)),
                                      ("PhotonVotingClassifier", PipelineElement("PhotonVotingClassifier"))])

        copy_of_the_pipe = photon_pipe.copy_me()

        self.assertEqual(photon_pipe.random_state, copy_of_the_pipe.random_state)
        self.assertTrue(len(copy_of_the_pipe.elements) == 5)
        self.assertTrue(copy_of_the_pipe.elements[2][1].name == "RandomStack")
        self.assertTrue(copy_of_the_pipe.named_steps["my_copy_switch"].elements[1].test_disabled)
        self.assertDictEqual(copy_of_the_pipe.elements[2][1].elements[1].elements[0].hyperparameters,
                             {"PCA__n_components": [5, 10]})
        self.assertTrue(isinstance(copy_of_the_pipe.elements[3][1], CallbackElement))
        self.assertTrue(copy_of_the_pipe.named_steps["tmp_callback"].delegate_function == np.mean)

    def test_random_state(self):
        photon_pipe = PhotonPipeline([("SS", self.p_ss), ("PCA", PipelineElement('PCA')), ("SVC", self.p_dt)])
        photon_pipe.random_state = 666
        photon_pipe.fit(self.X, self.y)
        self.assertEqual(self.p_dt.random_state, photon_pipe.random_state)
        self.assertEqual(photon_pipe.elements[1][-1].random_state, photon_pipe.random_state)
        self.assertEqual(self.p_dt.random_state, 666)


class CacheManagerTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(CacheManagerTests, cls).setUpClass()

    def setUp(self):
        super(CacheManagerTests, self).setUp()

        self.cache_man = CacheManager("123353423434", self.cache_folder_path)
        self.X, self.y, self.kwargs = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), \
                                      {'covariates': [9, 8, 7, 6, 5]}

        self.config1 = {'PCA__n_components': 5,
                        'SVC__C': 3,
                        'SVC__kernel': 'rbf'}
        self.item_names = ["StandardScaler", "PCA", "SVC"]

        self.config2 = {'PCA__n_components': 20,
                        'SVC__C': 1,
                        'SVC__kernel': 'linear'}

    def test_find_relevant_configuration_items(self):
        self.cache_man.prepare(pipe_elements=self.item_names, X=self.X, config=self.config1)
        relevant_items = {'PCA__n_components': 5}
        relevant_items_hash = hash(frozenset(relevant_items.items()))
        new_hash = self.cache_man._find_config_for_element("PCA")
        self.assertEqual(relevant_items_hash, new_hash)

    def test_empty_config(self):
        self.cache_man.prepare(pipe_elements=self.item_names, X=self.X, config={})
        relevant_items_hash = hash(frozenset({}.items()))
        new_hash = self.cache_man._find_config_for_element("PCA")
        self.assertEqual(relevant_items_hash, new_hash)

    def test_initial_transformation(self):
        self.cache_man.prepare(pipe_elements=self.item_names, config=self.config1)
        result = self.cache_man.load_cached_data("PCA")
        self.assertEqual(result, None)

    def test_check_cache(self):
        self.cache_man.prepare(pipe_elements=self.item_names, config=self.config1)
        self.assertFalse(self.cache_man.check_cache("PCA"))
        self.cache_man.save_data_to_cache("PCA", (self.X, self.y, self.kwargs))
        self.assertTrue(self.cache_man.check_cache("PCA"))

    def test_key_hash_equal(self):
        self.cache_man.prepare(pipe_elements=self.item_names, config=self.config1)
        generator_1 = self.cache_man.generate_cache_key("PCA")
        generator_2 = self.cache_man.generate_cache_key("PCA")
        self.assertEqual(generator_1, generator_2)

    def test_saving_and_loading_transformation(self):
        self.cache_man.prepare(pipe_elements=self.item_names, config=self.config1)
        self.cache_man.save_data_to_cache("PCA", (self.X, self.y, self.kwargs))

        self.assertTrue(len(self.cache_man.cache_index) == 1)
        for hash_key, cache_file in self.cache_man.cache_index.items():
            self.assertTrue(os.path.isfile(cache_file))

        result = self.cache_man.load_cached_data("PCA")
        self.assertTrue(result is not None)
        X_loaded, y_loaded, kwargs_loaded = result[0], result[1], result[2]
        self.assertTrue(np.array_equal(self.X, X_loaded))
        self.assertTrue(np.array_equal(self.y, y_loaded))
        self.assertTrue(np.array_equal(self.kwargs['covariates'], kwargs_loaded['covariates']))

    def test_clearing_folder(self):
        self.cache_man.clear_cache()
        self.assertTrue(len(glob.glob(os.path.join(self.cache_man.cache_folder, "*.p"))) == 0)


class CachedPhotonPipelineTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(CachedPhotonPipelineTests, cls).setUpClass()

    def setUp(self):
        super(CachedPhotonPipelineTests, self).setUp()
        # Photon Version
        ss = PipelineElement("StandardScaler", {})
        pca = PipelineElement("PCA", {'n_components': [3, 10, 50]}, random_state=3)
        svm = PipelineElement("SVC", {'kernel': ['rbf', 'linear']}, random_state=3)

        self.pipe = PhotonPipeline([('StandardScaler', ss),
                                    ('PCA', pca),
                                    ('SVC', svm)])

        self.pipe.caching = True
        self.pipe.fold_id = "12345643463434"
        CacheManager.clear_cache_files(self.cache_folder_path)
        self.pipe.cache_folder = self.cache_folder_path

        self.config1 = {'PCA__n_components': 4,
                        'SVC__C': 3,
                        'SVC__kernel': 'rbf'}

        self.config2 = {'PCA__n_components': 7,
                        'SVC__C': 1,
                        'SVC__kernel': 'linear'}

        self.X, self.y = load_breast_cancer(return_X_y=True)

    def test_group_caching(self):

        # transform one config
        self.pipe.set_params(**self.config1)
        self.pipe.fit(self.X, self.y)
        X_new, y_new, kwargs_new = self.pipe.transform(self.X, self.y)
        # one result should be cached ( one standard scaler output + one pca output)
        self.assertTrue(len(glob.glob(os.path.join(self.pipe.cache_folder, "*.p"))) == 2)

        # transform second config
        self.pipe.set_params(**self.config2)
        self.pipe.fit(self.X, self.y)
        X_config2, y_config2, kwargs_config2 = self.pipe.transform(self.X, self.y)
        # two results should be cached ( one standard scaler output (config hasn't changed)
        # + two pca outputs  )
        self.assertTrue(len(glob.glob(os.path.join(self.pipe.cache_folder, "*.p"))) == 3)

        # now transform with config 1 again, results should be loaded
        self.pipe.set_params(**self.config1)
        self.pipe.fit(self.X, self.y)
        X_2, y_2, kwargs_2 = self.pipe.transform(self.X, self.y)
        self.assertTrue(np.array_equal(X_new, X_2))
        self.assertTrue(np.array_equal(y_new, y_2))
        self.assertTrue(np.array_equal(kwargs_new, kwargs_2))

        # results should be the same as when caching is deactivated
        self.pipe.caching = False
        self.pipe.set_params(**self.config1)
        self.pipe.fit(self.X, self.y)
        X_uc, y_uc, kwargs_uc = self.pipe.transform(self.X, self.y)
        self.assertTrue(np.array_equal(X_uc, X_2))
        self.assertTrue(np.array_equal(y_uc, y_2))
        self.assertTrue(np.array_equal(kwargs_uc, kwargs_2))

    def test_empty_hyperparameters(self):
        # test if one can use it when only default parameters are given and hyperparameter space is empty
        self.pipe.set_params(**{})
        self.pipe.fit(self.X, self.y)
        X_new, y_new, kwargs_new = self.pipe.transform(self.X, self.y)
        # one result should be cached ( one standard scaler output + one pca output )
        self.assertTrue(len(glob.glob(os.path.join(self.pipe.cache_folder, "*.p"))) == 2)

        self.pipe.set_params(**{})
        self.pipe.fit(self.X, self.y)
        X_new2, y_new2, kwargs_new2 = self.pipe.transform(self.X, self.y)
        # assert nothing happened in the cache folder
        self.assertTrue(len(glob.glob(os.path.join(self.pipe.cache_folder, "*.p"))) == 2)
        self.assertTrue(np.array_equal(X_new, X_new2))
        self.assertTrue(np.array_equal(y_new, y_new2))
        self.assertTrue(np.array_equal(kwargs_new, kwargs_new2))

