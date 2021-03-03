import unittest
import warnings
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from photonai.base import PipelineElement, Switch, Stack, Branch, DataFilter, CallbackElement, Preprocessing
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.helper.helper import PhotonDataHelper
from photonai.helper.dummy_elements import DummyEstimator, \
    DummyNeedsCovariatesEstimator, DummyNeedsCovariatesTransformer, DummyNeedsYTransformer, DummyTransformer, \
    DummyNeedsCovariatesAndYTransformer, DummyEstimatorNoPredict, DummyEstimatorWrongType, DummyTransformerWithPredict
from photonai.helper.photon_base_test import elements_to_dict
from photonai.optimization import GridSearchOptimizer


class PipelineElementTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True, random_state=42)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']}, random_state=42)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.kwargs = {'covariates': self.y}
        self.Xt = self.X + 1
        self.yt = self.y + 1
        self.kwargst = {'covariates': self.y + 1}

    def test_create_failure(self):
        with self.assertRaises(NameError):
            PipelineElement('NONSENSEName', {})

    def test_pipeline_element_create(self):
        # test name, set_disabled and base_element
        self.assertIsInstance(self.pca_pipe_element.base_element, PCA)

        # set_disabled is passed correctly
        self.assertTrue(self.pca_pipe_element.test_disabled)

        # correct name
        self.assertEqual(self.pca_pipe_element.name, 'PCA')

    def test_create_from_object(self):
        pca = PipelineElement.create('obj_name', PCA(), hyperparameters={'n_components': [1, 4]})
        self.assertIsInstance(pca.base_element, PCA)
        self.assertEqual(pca.name, 'obj_name')

        with self.assertRaises(ValueError):
            pca_2 = PipelineElement.create('obj_name', PCA, hyperparameters={'n_components': [1, 4]})

    def test_fit_and_score(self):
        tmp_pca = PCA().fit(self.X, self.y)
        self.pca_pipe_element.fit(self.X, self.y)
        self.assertEqual(self.pca_pipe_element.base_element.components_.shape, (30, 30))
        self.assertAlmostEqual(self.pca_pipe_element.base_element.components_[0, 0], tmp_pca.components_[0, 0])

        tmp_svc = SVC().fit(self.X, self.y)
        self.svc_pipe_element.fit(self.X, self.y)
        self.assertAlmostEqual(self.svc_pipe_element.base_element._intercept_[0], tmp_svc._intercept_[0])

        sk_score = tmp_svc.score(self.X, self.y)
        p_score = self.svc_pipe_element.score(self.X, self.y)
        self.assertTrue(np.array_equal(sk_score, p_score))

    def test_transform(self):
        self.pca_pipe_element.fit(self.X, self.y)

        Xt, _, _ = self.pca_pipe_element.transform(self.X)
        self.assertEqual(Xt.shape, (569, 30))
        self.assertAlmostEqual(Xt[0, 0], 1160.1425737041347)

    def test_predict(self):
        self.svc_pipe_element.fit(self.X, self.y)

        yt = self.svc_pipe_element.predict(self.X)
        self.assertEqual(yt.shape, (569,))
        self.assertEqual(yt[21], 1)

    def test_predict_proba(self):
        self.svc_pipe_element.fit(self.X, self.y)
        self.assertEqual(self.svc_pipe_element.predict_proba(self.X), None)

        gpc = PipelineElement('GaussianProcessClassifier')
        gpc.fit(self.X, self.y)
        self.assertTrue(np.array_equal(gpc.predict_proba(self.X)[0], np.asarray([0.5847072926551391, 0.4152927073448609])))

    def test_inverse_transform(self):
        Xt, _, _ = self.pca_pipe_element.fit(self.X, self.y).transform(self.X)
        X, _, _ = self.pca_pipe_element.inverse_transform(Xt)
        np.testing.assert_array_almost_equal(X, self.X)

    def test_one_hyperparameter_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.pca_pipe_element.hyperparameters, {'PCA__n_components': [1, 2],
                                                                     'PCA__disabled': [False, True]})

        # config_grid is created as expected
        self.assertListEqual(self.pca_pipe_element.generate_config_grid(), [{'PCA__n_components': 1,
                                                                             'PCA__disabled': False},
                                                                            {'PCA__n_components': 2,
                                                                             'PCA__disabled': False},
                                                                            {'PCA__disabled': True}])

    def test_hyperprameter_sanity_check(self):
        with self.assertRaises(ValueError):
            error_element = PipelineElement('PCA', hyperparameters={'kernel': ['rbf', 'linear']})

    def test_more_hyperparameters_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.svc_pipe_element.hyperparameters, {'SVC__C': [0.1, 1],
                                                                     'SVC__kernel': ['rbf', 'sigmoid']})

        # config_grid is created as expected
        self.assertListEqual(self.svc_pipe_element.generate_config_grid(), [{'SVC__C': 0.1, 'SVC__kernel': 'rbf'},
                                                                            {'SVC__C': 0.1, 'SVC__kernel': 'sigmoid'},
                                                                            {'SVC__C': 1, 'SVC__kernel': 'rbf'},
                                                                            {'SVC__C': 1, 'SVC__kernel': 'sigmoid'}])

    def test_no_hyperparameters(self):
        pca_sklearn_element = PCA()
        pca_photon_element = PipelineElement('PCA')

        self.assertDictEqual(pca_sklearn_element.__dict__, pca_photon_element.base_element.__dict__)

    def test_set_params(self):
        config = {'n_components': 3, 'disabled': False}
        self.pca_pipe_element.set_params(**config)
        self.assertFalse(self.pca_pipe_element.disabled)
        self.assertEqual(self.pca_pipe_element.base_element.n_components, 3)
        with self.assertRaises(ValueError):
            self.pca_pipe_element.set_params(**{'any_weird_param': 1})

    def test_get_params(self):
        photon_params = PipelineElement('PCA').get_params()
        del photon_params["name"]
        sk_params = PCA().get_params()
        self.assertDictEqual(photon_params, sk_params)
        self.assertIsNone(PipelineElement.create('any_not_existing_object', object(), hyperparameters={}).get_params())

    def test_set_random_state(self):
        # we handle all elements in one method that is inherited so we capture them all in this test
        random_state = 53
        my_branch = Branch("random_state_branch")
        my_branch += PipelineElement("StandardScaler")
        my_switch = Switch("transformer_Switch")
        my_switch += PipelineElement("LassoFeatureSelection")
        my_switch += PipelineElement("PCA")
        my_branch += my_switch
        my_stack = Stack("Estimator_Stack")
        my_stack += PipelineElement("SVR")
        my_stack += PipelineElement("Ridge")
        my_branch += my_stack
        my_branch += PipelineElement("ElasticNet")

        my_branch.random_state = random_state
        self.assertTrue(my_switch.elements[1].random_state == random_state)
        self.assertTrue(my_switch.elements[1].base_element.random_state == random_state)
        self.assertTrue(my_stack.elements[1].random_state == random_state)
        self.assertTrue(my_stack.elements[1].base_element.random_state == random_state)

    def test_adjusted_delegate_call_transformer(self):
        # check standard transformer
        trans = PipelineElement.create('Transformer', base_element=DummyTransformer(), hyperparameters={})
        X, y, kwargs = trans.transform(self.X, self.y, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))  # only X should be transformed
        self.assertTrue(np.array_equal(y, self.y))
        self.assertDictEqual(kwargs, self.kwargs)

        # check transformer needs y
        trans = PipelineElement.create('NeedsYTransformer', base_element=DummyNeedsYTransformer(), hyperparameters={})
        X, y, kwargs = trans.transform(self.X, self.y, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))
        self.assertTrue(np.array_equal(y, self.yt))
        self.assertDictEqual(kwargs, self.kwargs)

        trans = PipelineElement.create('NeedsYTransformer', base_element=DummyNeedsYTransformer(), hyperparameters={})
        X, y, kwargs = trans.transform(self.X, self.y)  # this time without any kwargs
        self.assertTrue(np.array_equal(X, self.Xt))
        self.assertTrue(np.array_equal(y, self.yt))
        self.assertDictEqual(kwargs, {})

        # check transformer needs covariates
        trans = PipelineElement.create('NeedsCovariatesTransformer', base_element=DummyNeedsCovariatesTransformer(),
                                       hyperparameters={})
        X, y, kwargs = trans.transform(self.X, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))
        self.assertTrue(np.array_equal(kwargs['covariates'], self.kwargst['covariates']))
        self.assertEqual(y, None)

        # check transformer needs covariates and needs y
        trans = PipelineElement.create('NeedsCovariatesAndYTransformer', base_element=DummyNeedsCovariatesAndYTransformer(),
                                       hyperparameters={})
        X, y, kwargs = trans.transform(self.X, self.y, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))
        self.assertTrue(np.array_equal(y, self.yt))
        self.assertTrue(np.array_equal(kwargs['covariates'], self.kwargst['covariates']))

    def test_adjusted_delegate_call_estimator(self):
        # check standard estimator
        est = PipelineElement.create('Estimator', base_element=DummyEstimator(), hyperparameters={})
        y = est.predict(self.X)
        self.assertTrue(np.array_equal(y, self.Xt)) # DummyEstimator returns X as y predictions

        # check estimator needs covariates
        est = PipelineElement.create('Estimator', base_element=DummyNeedsCovariatesEstimator(), hyperparameters={})
        X = est.predict(self.X, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))  # DummyEstimator returns X as y predictions

    def test_predict_when_no_transform(self):
        # check standard estimator
        est = PipelineElement.create('Estimator', base_element=DummyEstimator(), hyperparameters={})
        X, y, kwargs = est.transform(self.X)
        self.assertTrue(np.array_equal(X, self.Xt))  # DummyEstimator returns X as y predictions
        self.assertEqual(y, None)

        # check estimator needs covariates
        est = PipelineElement.create('Estimator', base_element=DummyNeedsCovariatesEstimator(), hyperparameters={})
        X, y, kwargs = est.transform(self.X, **self.kwargs)
        self.assertTrue(np.array_equal(X, self.Xt))  # DummyEstimator returns X as y predictions
        self.assertTrue(np.array_equal(kwargs['covariates'], self.kwargs['covariates']))
        self.assertEqual(y, None)

    def test_predict_on_transformer(self):
        est = PipelineElement.create('Estimator', base_element=DummyTransformer(), hyperparameters={})
        with self.assertRaises(BaseException):
            est.predict(self.X)

    def test_copy_me(self):
        svc = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        svc.set_params(**{'C': 0.1, 'kernel': 'sigmoid'})
        copy = svc.copy_me()

        self.assertEqual(svc.random_state, copy.random_state)
        self.assertNotEqual(copy.base_element, svc.base_element)
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(svc))
        self.assertEqual(copy.base_element.C, svc.base_element.C)

        # check if copies are still the same, even when making a copy of a fitted PipelineElement
        copy_after_fit = svc.fit(self.X, self.y).copy_me()
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(copy_after_fit))

        svc = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        copy = svc.copy_me()
        self.assertDictEqual(copy.hyperparameters, {'SVC__C': [0.1, 1], 'SVC__kernel': ['rbf', 'sigmoid']})
        copy.base_element.C = 3
        self.assertNotEqual(svc.base_element.C, copy.base_element.C)

        # test custom element
        custom_element = PipelineElement.create('CustomElement', base_element=DummyNeedsCovariatesEstimator(),
                                                hyperparameters={})
        copy = custom_element.copy_me()
        self.assertDictEqual(elements_to_dict(custom_element), elements_to_dict(copy))

        custom_element2 = PipelineElement.create('MyUnDeepcopyableObject', base_element=GridSearchOptimizer(),
                                                 hyperparameters={})
        with self.assertRaises(Exception):
            custom_element2.copy_me()

    def test_estimator_type(self):
        estimator = PipelineElement('SVC')
        self.assertEqual(estimator._estimator_type, 'classifier')

        estimator = PipelineElement('SVR')
        self.assertEqual(estimator._estimator_type, 'regressor')

        estimator = PipelineElement('PCA')
        self.assertEqual(estimator._estimator_type, None)

        estimator = PipelineElement.create('Dummy', DummyEstimatorWrongType(), {})
        with self.assertRaises(NotImplementedError):
            est_type = estimator._estimator_type

        estimator = PipelineElement.create('Dummy', DummyTransformerWithPredict(), {})
        with self.assertRaises(NotImplementedError):
            est_type = estimator._estimator_type

        estimator = PipelineElement.create('Dummy', DummyEstimatorNoPredict(), {})
        with self.assertRaises(NotImplementedError):
            est_type = estimator._estimator_type

    def test_sanity_check_item_for_add(self):
        valid_type = PipelineElement('StandardScaler')
        valid_type2 = CallbackElement('my_callback', None)
        invalid_type = StandardScaler()
        invalid_type2 = Preprocessing()

        PipelineElement.sanity_check_element_type_for_building_photon_pipes(valid_type, PipelineElement)
        PipelineElement.sanity_check_element_type_for_building_photon_pipes(valid_type2, PipelineElement)

        with self.assertRaises(TypeError):
            PipelineElement.sanity_check_element_type_for_building_photon_pipes(invalid_type, PipelineElement)

        with self.assertRaises(TypeError):
            PipelineElement.sanity_check_element_type_for_building_photon_pipes(invalid_type2, PipelineElement)

        classes_to_test = [Stack, Switch, Branch, Preprocessing]
        for photon_class in classes_to_test:
            # we name it SVC so it suits all classes
            if photon_class is Preprocessing:
                instance = photon_class()
            else:
                instance = photon_class('tmp_instance')
            instance.add(valid_type)
            instance.add(valid_type2)
            with self.assertRaises(TypeError):
                instance.add(invalid_type)
            with self.assertRaises(TypeError):
                instance.add(invalid_type2)


class SwitchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.svc = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.tree = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]})
        self.gpc = PipelineElement('GaussianProcessClassifier')
        self.pca = PipelineElement('PCA')

        self.estimator_branch = Branch('estimator_branch', [self.tree.copy_me()])
        self.transformer_branch = Branch('transformer_branch', [self.pca.copy_me()])

        self.estimator_switch = Switch('estimator_switch',
                                       [self.svc.copy_me(), self.tree.copy_me(), self.gpc.copy_me()])
        self.estimator_switch_with_branch = Switch('estimator_switch_with_branch',
                                                   [self.tree.copy_me(), self.estimator_branch.copy_me()])
        self.transformer_switch_with_branch = Switch('transformer_switch_with_branch',
                                                     [self.pca.copy_me(), self.transformer_branch.copy_me()])
        self.switch_in_switch = Switch('Switch_in_switch',
                                       [self.transformer_branch.copy_me(),
                                        self.transformer_switch_with_branch.copy_me()])

    def test_init(self):
        self.assertEqual(self.estimator_switch.name, 'estimator_switch')

    def test_hyperparams(self):
        # assert number of different configs to test
        # each config combi for each element: 4 for SVC and 3 for logistic regression = 7
        self.assertEqual(len(self.estimator_switch.pipeline_element_configurations), 3)
        self.assertEqual(len(self.estimator_switch.pipeline_element_configurations[0]), 4)
        self.assertEqual(len(self.estimator_switch.pipeline_element_configurations[1]), 3)

        # hyperparameters
        self.assertDictEqual(self.estimator_switch.hyperparameters,
                             {'estimator_switch__current_element': [(0, 0), (0, 1), (0, 2), (0, 3),
                                                                    (1, 0), (1, 1), (1, 2), (2, 0)]})

        # config grid
        self.assertListEqual(self.estimator_switch.generate_config_grid(),
                             [{'estimator_switch__current_element': (0, 0)},
                              {'estimator_switch__current_element': (0, 1)},
                              {'estimator_switch__current_element': (0, 2)},
                              {'estimator_switch__current_element': (0, 3)},
                              {'estimator_switch__current_element': (1, 0)},
                              {'estimator_switch__current_element': (1, 1)},
                              {'estimator_switch__current_element': (1, 2)},
                              {'estimator_switch__current_element': (2, 0)}])

    def test_set_params(self):

        # test for grid search
        false_config = {'current_element': 1}
        with self.assertRaises(ValueError):
            self.estimator_switch.set_params(**false_config)

        correct_config = {'current_element': (0, 1)}
        self.estimator_switch.set_params(**correct_config)
        self.assertEqual(self.estimator_switch.base_element.base_element.C, 0.1)
        self.assertEqual(self.estimator_switch.base_element.base_element.kernel, 'sigmoid')

        # test for other optimizers
        smac_config = {'SVC__C': 2, 'SVC__kernel': 'rbf'}
        self.estimator_switch.set_params(**smac_config)
        self.assertEqual(self.estimator_switch.base_element.base_element.C, 2)
        self.assertEqual(self.estimator_switch.base_element.base_element.kernel, 'rbf')

    def test_fit(self):
        np.random.seed(42)
        self.estimator_switch.set_params(**{'current_element': (1, 0)})
        self.estimator_switch.fit(self.X, self.y)
        np.random.seed(42)
        self.tree.set_params(**{'min_samples_split': 2})
        self.tree.fit(self.X, self.y)
        np.testing.assert_array_equal(self.tree.base_element.feature_importances_,
                                      self.estimator_switch.base_element.feature_importances_)

    def test_transform(self):
        self.transformer_switch_with_branch.set_params(**{'current_element': (0, 0)})
        self.transformer_switch_with_branch.fit(self.X, self.y)
        self.pca.fit(self.X, self.y)

        switch_Xt, _, _ = self.transformer_switch_with_branch.transform(self.X)
        pca_Xt, _, _ = self.pca.transform(self.X)
        self.assertTrue(np.array_equal(pca_Xt, switch_Xt))

    def test_predict(self):
        self.estimator_switch.set_params(**{'current_element': (1, 0)})
        np.random.seed(42)
        self.estimator_switch.fit(self.X, self.y)
        self.tree.set_params(**{'min_samples_split': 2})
        np.random.seed(42)
        self.tree.fit(self.X, self.y)

        switch_preds = self.estimator_switch.predict(self.X)
        tree_preds = self.tree.predict(self.X)
        self.assertTrue(np.array_equal(switch_preds, tree_preds))

    def test_predict_proba(self):
        gpc = PipelineElement('GaussianProcessClassifier')
        svc = PipelineElement('SVC')
        switch = Switch('EstimatorSwitch', [gpc, svc])
        switch.set_params(**{'current_element': (0, 0)})
        np.random.seed(42)
        switch_probas = switch.fit(self.X, self.y).predict_proba(self.X)
        np.random.seed(42)
        gpr_probas = self.gpc.fit(self.X, self.y).predict_proba(self.X)
        self.assertTrue(np.array_equal(switch_probas, gpr_probas))

    def test_inverse_transform(self):
        self.transformer_switch_with_branch.set_params(**{'current_element': (0, 0)})
        self.transformer_switch_with_branch.fit(self.X, self.y)
        self.pca.fit(self.X, self.y)
        Xt_pca, _, _ = self.pca.transform(self.X)
        Xt_switch, _, _ = self.transformer_switch_with_branch.transform(self.X)
        X_pca, _, _ = self.pca.inverse_transform(Xt_pca)
        X_switch, _, _ = self.transformer_switch_with_branch.inverse_transform(Xt_switch)

        self.assertTrue(np.array_equal(Xt_pca, Xt_switch))
        self.assertTrue(np.array_equal(X_pca, X_switch))
        np.testing.assert_almost_equal(X_switch, self.X)

    def test_base_element(self):
        switch = Switch('switch', [self.svc, self.tree])
        switch.set_params(**{'current_element': (1, 1)})
        self.assertIs(switch.base_element, self.tree)
        self.assertIs(switch.base_element.base_element, self.tree.base_element)

        # other optimizer
        switch.set_params(**{'DecisionTreeClassifier__min_samples_split': 2})
        self.assertIs(switch.base_element, self.tree)
        self.assertIs(switch.base_element.base_element, self.tree.base_element)

    def test_copy_me(self):
        switches = [self.estimator_switch, self.estimator_switch_with_branch, self.transformer_switch_with_branch,
                    self.switch_in_switch]

        for switch in switches:
            copy = switch.copy_me()

            self.assertEqual(switch.random_state, copy.random_state)

            for i, element in enumerate(copy.elements):
                self.assertNotEqual(copy.elements[i], switch.elements[i])

            switch = elements_to_dict(switch)
            copy = elements_to_dict(copy)

            self.assertDictEqual(copy, switch)

    def test_estimator_type(self):
        pca = PipelineElement('PCA')
        ica = PipelineElement('FastICA')
        svc = PipelineElement('SVC')
        svr = PipelineElement('SVR')
        tree_class = PipelineElement('DecisionTreeClassifier')
        tree_reg = PipelineElement('DecisionTreeRegressor')

        switch = Switch('MySwitch', [pca, svr])
        with self.assertRaises(NotImplementedError):
            est_type = switch._estimator_type

        switch = Switch('MySwitch', [svc, svr])
        with self.assertRaises(NotImplementedError):
            est_type = switch._estimator_type

        switch = Switch('MySwitch', [pca, ica])
        self.assertEqual(switch._estimator_type, None)

        switch = Switch('MySwitch', [tree_class, svc])
        self.assertEqual(switch._estimator_type, 'classifier')

        switch = Switch('MySwitch', [tree_reg, svr])
        self.assertEqual(switch._estimator_type, 'regressor')

        self.assertEqual(self.estimator_switch._estimator_type, 'classifier')
        self.assertEqual(self.estimator_switch_with_branch._estimator_type, 'classifier')
        self.assertEqual(self.transformer_switch_with_branch._estimator_type, None)
        self.assertEqual(self.switch_in_switch._estimator_type, None)

    def test_add(self):
        self.assertEqual(len(self.estimator_switch.elements), 3)
        self.assertEqual(len(self.switch_in_switch.elements), 2)
        self.assertEqual(len(self.transformer_switch_with_branch.elements), 2)

        self.assertEqual(list(self.estimator_switch.elements_dict.keys()), ['SVC', 'DecisionTreeClassifier',
                                                                            'GaussianProcessClassifier'])
        self.assertEqual(list(self.switch_in_switch.elements_dict.keys()), ['transformer_branch',
                                                                            'transformer_switch_with_branch'])

        switch = Switch('MySwitch', [PipelineElement('PCA'), PipelineElement('FastICA')])
        switch = Switch('MySwitch2')
        switch += PipelineElement('PCA')
        switch += PipelineElement('FastICA')

        # test doubled names
        with self.assertRaises(ValueError):
            self.estimator_switch += self.estimator_switch.elements[0]
        self.estimator_switch += PipelineElement("SVC")
        self.assertEqual(self.estimator_switch.elements[-1].name, "SVC2")
        self.estimator_switch += PipelineElement("SVC", hyperparameters={'kernel': ['polynomial', 'sigmoid']})
        self.assertEqual(self.estimator_switch.elements[-1].name, "SVC3")
        self.estimator_switch += PipelineElement("SVR")
        self.assertEqual(self.estimator_switch.elements[-1].name, "SVR")
        self.estimator_switch += PipelineElement("SVC")
        self.assertEqual(self.estimator_switch.elements[-1].name, "SVC4")

        # check that hyperparameters are renamed respectively
        self.assertEqual(self.estimator_switch.pipeline_element_configurations[4][0]["SVC3__kernel"], 'polynomial')

    def test_feature_importances(self):

        self.estimator_switch.set_params(**{'current_element': (1, 0)})
        self.estimator_switch.fit(self.X, self.y)
        self.assertTrue(len(self.estimator_switch.feature_importances_) == self.X.shape[1])

        self.estimator_switch_with_branch.set_params(**{'current_element': (1, 0)})
        self.estimator_switch_with_branch.fit(self.X, self.y)
        self.assertTrue(len(self.estimator_switch_with_branch.feature_importances_) == self.X.shape[1])

        self.estimator_switch.set_params(**{'current_element': (2, 0)})
        self.estimator_switch.fit(self.X, self.y)
        self.assertIsNone(self.estimator_branch.feature_importances_)

        self.switch_in_switch.set_params(**{'current_element': (1, 0)})
        self.switch_in_switch.fit(self.X, self.y)
        self.assertIsNone(self.switch_in_switch.feature_importances_)
        self.estimator_switch.set_params(**{'current_element': (1, 0)})
        self.switch_in_switch.fit(self.X, self.y)
        self.assertIsNone(self.switch_in_switch.feature_importances_)


class BranchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.scaler = PipelineElement("StandardScaler", {'with_mean': True})
        self.pca = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True, random_state=3)
        self.tree = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]}, random_state=3)

        self.transformer_branch = Branch('MyBranch', [self.scaler, self.pca])
        self.transformer_branch_sklearn = SKPipeline([("SS", StandardScaler()),
                                                      ("PCA", PCA(random_state=3))])
        self.estimator_branch = Branch('MyBranch', [self.scaler, self.pca, self.tree])
        self.estimator_branch_sklearn = SKPipeline([("SS", StandardScaler()),
                                                    ("PCA", PCA(random_state=3)),
                                                    ("Tree", DecisionTreeClassifier(random_state=3))])

    def test_fit(self):
        self.estimator_branch_sklearn.fit(self.X, self.y)
        sk_pred = self.estimator_branch_sklearn.predict(self.X)

        self.estimator_branch.fit(self.X, self.y)
        branch_pred = self.estimator_branch.predict(self.X)

        self.assertTrue(np.array_equal(sk_pred, branch_pred))

    def test_transform(self):
        Xt, _, _ = self.transformer_branch.fit(self.X, self.y).transform(self.X)
        Xt_sklearn = self.transformer_branch_sklearn.fit(self.X, self.y).transform(self.X)
        self.assertTrue(np.array_equal(Xt, Xt_sklearn))

    def test_predict(self):
        y_pred = self.estimator_branch.fit(self.X, self.y).predict(self.X)
        y_pred_sklearn = self.estimator_branch_sklearn.fit(self.X, self.y).predict(self.X)
        np.testing.assert_array_equal(y_pred, y_pred_sklearn)

    def test_predict_proba(self):
        proba = self.estimator_branch.fit(self.X, self.y).predict_proba(self.X)
        proba_sklearn = self.estimator_branch_sklearn.fit(self.X, self.y).predict_proba(self.X)
        np.testing.assert_array_equal(proba, proba_sklearn)

    def test_inverse_transform(self):
        self.estimator_branch.fit(self.X, self.y)
        feature_importances = self.estimator_branch.elements[-1].base_element.feature_importances_
        Xt, _, _ = self.estimator_branch.inverse_transform(feature_importances)
        self.assertEqual(self.X.shape[1], Xt.shape[0])

    def test_no_y_transformers(self):
        stacking_element = Stack("forbidden_stack")
        my_dummy = PipelineElement.create("dummy", DummyNeedsCovariatesAndYTransformer(), {})

        with self.assertRaises(NotImplementedError):
            stacking_element += my_dummy

    def test_copy_me(self):
        branch = Branch('MyBranch', [self.scaler, self.pca])

        copy = branch.copy_me()
        self.assertEqual(branch.random_state, copy.random_state)
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(branch))

        copy = branch.copy_me()
        copy.elements[1].base_element.n_components = 3
        self.assertNotEqual(copy.elements[1].base_element.n_components, branch.elements[1].base_element.n_components)

        fake_copy = branch
        fake_copy.elements[1].base_element.n_components = 3
        self.assertEqual(fake_copy.elements[1].base_element.n_components, branch.elements[1].base_element.n_components)

    def test_prepare_photon_pipeline(self):
        test_branch = Branch('my_test_branch')
        test_branch += PipelineElement('SimpleImputer')
        test_branch += Switch('my_crazy_switch_bitch')
        test_branch += Stack('my_stacking_stack')
        test_branch += PipelineElement('SVC')

        generated_pipe = test_branch.prepare_photon_pipe(test_branch.elements)

        self.assertEqual(len(generated_pipe.named_steps), 4)
        for idx, element in enumerate(test_branch.elements):
            self.assertIs(generated_pipe.named_steps[element.name], element)
            self.assertIs(generated_pipe.elements[idx][1], test_branch.elements[idx])

    def test_sanity_check_pipe(self):
        test_branch = Branch('my_test_branch')

        def callback_func(X, y, **kwargs):
            pass

        with warnings.catch_warnings(record=True) as w:
            my_callback = CallbackElement('final_element_callback', delegate_function=callback_func)
            test_branch += my_callback
            no_callback_pipe = test_branch.prepare_photon_pipe(test_branch.elements)
            self.assertTrue(no_callback_pipe.elements[-1][1] is my_callback)
            test_branch.sanity_check_pipeline(no_callback_pipe)
            self.assertFalse(no_callback_pipe.elements)
            assert any("Last element of pipeline cannot be callback" in s for s in [e.message.args[0] for e in w])

    def test_prepare_pipeline(self):
        self.assertEqual(len(self.transformer_branch.elements), 2)
        config_grid = {'MyBranch__PCA__n_components': [1, 2],
                       'MyBranch__PCA__disabled': [False, True],
                       'MyBranch__StandardScaler__with_mean': True}
        self.assertDictEqual(config_grid, self.transformer_branch._hyperparameters)

    def test_set_params(self):
        config = {'PCA__n_components': 2,
                  'PCA__disabled': True,
                  'StandardScaler__with_mean': True}
        self.transformer_branch.set_params(**config)
        self.assertTrue(self.transformer_branch.base_element.elements[1][1].disabled)
        self.assertEqual(self.transformer_branch.base_element.elements[1][1].base_element.n_components, 2)
        self.assertEqual(self.transformer_branch.base_element.elements[0][1].base_element.with_mean, True)

        with self.assertRaises(ValueError):
            self.transformer_branch.set_params(**{'any_weird_param': 1})

    def test_estimator_type(self):
        def callback(X, y=None):
            pass

        transformer_branch = Branch('TransBranch', [PipelineElement('PCA'), PipelineElement('FastICA')])
        classifier_branch = Branch('ClassBranch', [PipelineElement('SVC')])
        regressor_branch = Branch('RegBranch', [PipelineElement('SVR')])
        callback_branch = Branch('CallBranch', [PipelineElement('SVR'), CallbackElement('callback', callback)])

        self.assertEqual(transformer_branch._estimator_type, None)
        self.assertEqual(classifier_branch._estimator_type, 'classifier')
        self.assertEqual(regressor_branch._estimator_type, 'regressor')
        self.assertEqual(callback_branch._estimator_type, None)

    def test_add(self):
        branch = Branch('MyBranch', [PipelineElement('PCA', {'n_components': [5]}), PipelineElement('FastICA')])
        self.assertEqual(len(branch.elements), 2)
        self.assertDictEqual(branch._hyperparameters, {'MyBranch__PCA__n_components': [5]})
        branch = Branch('MyBranch')
        branch += PipelineElement('PCA', {'n_components': [5]})
        branch += PipelineElement('FastICA')
        self.assertEqual(len(branch.elements), 2)
        self.assertDictEqual(branch._hyperparameters, {'MyBranch__PCA__n_components': [5]})

        # add doubled item
        branch += PipelineElement('PCA', {'n_components': [10, 20]})
        self.assertEqual(branch.elements[-1].name, 'PCA2')
        self.assertDictEqual(branch.hyperparameters, {'MyBranch__PCA__n_components': [5], 'MyBranch__PCA2__n_components': [10, 20]})

    def test_feature_importances(self):

        self.estimator_branch.fit(self.X, self.y)
        self.assertTrue(len(self.estimator_branch.feature_importances_) == self.X.shape[1])

        self.estimator_branch.elements[-1] = PipelineElement("GaussianProcessClassifier")
        self.estimator_branch.fit(self.X, self.y)
        self.assertIsNone(self.estimator_branch.feature_importances_)

        self.transformer_branch.fit(self.X, self.y)
        self.assertIsNone(self.transformer_branch.feature_importances_)


class StackTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)

        self.pca = PipelineElement('PCA', {'n_components': [5, 10]})
        self.scaler = PipelineElement('StandardScaler', {'with_mean': [True]})
        self.svc = PipelineElement('SVC', {'C': [1, 2]})
        self.tree = PipelineElement('DecisionTreeClassifier', {'min_samples_leaf': [3, 5]})

        self.transformer_branch_1 = Branch('TransBranch1', [self.pca.copy_me()])
        self.transformer_branch_2 = Branch('TransBranch2', [self.scaler.copy_me()])

        self.estimator_branch_1 = Branch('EstBranch1', [self.svc.copy_me()])
        self.estimator_branch_2 = Branch('EstBranch2', [self.tree.copy_me()])

        self.transformer_stack = Stack('TransformerStack', [self.pca.copy_me(), self.scaler.copy_me()])
        self.estimator_stack = Stack('EstimatorStack', [self.svc.copy_me(), self.tree.copy_me()])
        self.transformer_branch_stack = Stack('TransBranchStack', [self.transformer_branch_1.copy_me(),
                                                                   self.transformer_branch_2.copy_me()])
        self.estimator_branch_stack = Stack('EstBranchStack', [self.estimator_branch_1.copy_me(),
                                                               self.estimator_branch_2.copy_me()])

        self.stacks = [([self.pca, self.scaler], self.transformer_stack),
                       ([self.svc, self.tree], self.estimator_stack),
                       ([self.transformer_branch_1, self.transformer_branch_2], self.transformer_branch_stack),
                       ([self.estimator_branch_1, self.estimator_branch_2], self.estimator_branch_stack)]

    def test_copy_me(self):
        for stack in self.stacks:
            stack = stack[1]
            copy = stack.copy_me()
            self.assertEqual(stack.random_state, copy.random_state)
            self.assertFalse(stack.elements[0].__dict__ == copy.elements[0].__dict__)
            self.assertDictEqual(elements_to_dict(stack), elements_to_dict(copy))

    def test_horizontal_stacking(self):
        for stack in self.stacks:
            element_1 = stack[0][0]
            element_2 = stack[0][1]
            stack = stack[1]

            # fit elements
            Xt_1 = element_1.fit(self.X, self.y).transform(self.X, self.y)
            Xt_2 = element_2.fit(self.X, self.y).transform(self.X, self.y)

            Xt = stack.fit(self.X, self.y).transform(self.X, self.y)

            # output of transform() changes depending on whether it is an estimator stack or a transformer stack
            if isinstance(Xt, tuple):
                Xt = Xt[0]
                Xt_1 = Xt_1[0]
                Xt_2 = Xt_2[0]

            if len(Xt_1.shape) == 1:
                Xt_1 = np.reshape(Xt_1, (-1, 1))
                Xt_2 = np.reshape(Xt_2, (-1, 1))

            self.assertEqual(Xt.shape[1], Xt_1.shape[-1] + Xt_2.shape[-1])

    def recursive_assertion(self, element_a, element_b):
        for key in element_a.keys():
            if isinstance(element_a[key], np.ndarray):
                np.testing.assert_array_equal(element_a[key], element_b[key])
            elif isinstance(element_a[key], dict):
                self.recursive_assertion(element_a[key], element_b[key])
            else:
                self.assertEqual(element_a[key], element_b[key])

    def test_fit(self):
        for elements, stack in [([self.pca, self.scaler], self.transformer_stack),
                                ([self.svc, self.tree], self.estimator_stack)]:
            np.random.seed(42)
            stack = stack.fit(self.X, self.y)
            np.random.seed(42)
            for i, element in enumerate(elements):
                element = element.fit(self.X, self.y)
                element_dict = elements_to_dict(element)
                stack_dict = elements_to_dict(stack.elements[i])
                self.recursive_assertion(element_dict, stack_dict)

    def test_transform(self):
        for elements, stack in self.stacks:
            np.random.seed(42)
            Xt_stack, _, _ = stack.fit(self.X, self.y).transform(self.X)
            np.random.seed(42)
            Xt_elements = None
            for i, element in enumerate(elements):
                Xt_element, _, _ = element.fit(self.X, self.y).transform(self.X)
                Xt_elements = PhotonDataHelper.stack_data_horizontally(Xt_elements, Xt_element)
            np.testing.assert_array_equal(Xt_stack, Xt_elements)

    def test_predict(self):
        for elements, stack in [([self.svc, self.tree], self.estimator_stack),
                                ([self.estimator_branch_1, self.estimator_branch_2], self.estimator_branch_stack)]:
            np.random.seed(42)
            stack = stack.fit(self.X, self.y)
            yt_stack = stack.predict(self.X)
            np.random.seed(42)
            Xt_elements = None
            for i, element in enumerate(elements):
                Xt_element = element.fit(self.X, self.y).predict(self.X)
                Xt_elements = PhotonDataHelper.stack_data_horizontally(Xt_elements, Xt_element)
            np.testing.assert_array_equal(yt_stack, Xt_elements)

    def test_predict_proba(self):
        for elements, stack in [([self.svc, self.tree], self.estimator_stack),
                                ([self.estimator_branch_1, self.estimator_branch_2], self.estimator_branch_stack)]:
            np.random.seed(42)
            stack = stack.fit(self.X, self.y)
            yt_stack = stack.predict_proba(self.X)
            np.random.seed(42)
            Xt_elements = None
            for i, element in enumerate(elements):
                Xt_element = element.fit(self.X, self.y).predict_proba(self.X)
                if Xt_element is None:
                    Xt_element = element.fit(self.X, self.y).predict(self.X)
                Xt_elements = PhotonDataHelper.stack_data_horizontally(Xt_elements, Xt_element)
            np.testing.assert_array_equal(yt_stack, Xt_elements)

    def test_inverse_transform(self):
        with self.assertRaises(NotImplementedError):
            self.stacks[0][1].fit(self.X, self.y).inverse_transform(self.X)

    def test_set_params(self):
        trans_config = {'PCA__n_components': 2,
                        'PCA__disabled': True,
                        'StandardScaler__with_mean': True}
        est_config = {'SVC__C': 3,
                      'DecisionTreeClassifier__min_samples_leaf': 1}

        # transformer stack
        self.transformer_stack.set_params(**trans_config)
        self.assertEqual(self.transformer_stack.elements[0].base_element.n_components, 2)
        self.assertEqual(self.transformer_stack.elements[0].disabled, True)
        self.assertEqual(self.transformer_stack.elements[1].base_element.with_mean, True)

        # estimator stack
        self.estimator_stack.set_params(**est_config)
        self.assertEqual(self.estimator_stack.elements[0].base_element.C, 3)
        self.assertEqual(self.estimator_stack.elements[1].base_element.min_samples_leaf, 1)

        with self.assertRaises(ValueError):
            self.estimator_stack.set_params(**{'any_weird_param': 1})

        with self.assertRaises(ValueError):
            self.transformer_stack.set_params(**{'any_weird_param': 1})

    def test_add(self):
        stack = Stack('MyStack', [PipelineElement('PCA', {'n_components': [5]}), PipelineElement('FastICA')])
        self.assertEqual(len(stack.elements), 2)
        self.assertDictEqual(stack._hyperparameters, {'MyStack__PCA__n_components': [5]})
        stack = Stack('MyStack')
        stack += PipelineElement('PCA', {'n_components': [5]})
        stack += PipelineElement('FastICA')
        self.assertEqual(len(stack.elements), 2)
        self.assertDictEqual(stack._hyperparameters, {'MyStack__PCA__n_components': [5]})

        def callback(X, y=None):
            pass

        stack = Stack('MyStack', [PipelineElement('PCA'),
                                  CallbackElement('MyCallback', callback),
                                  Switch('MySwitch', [PipelineElement('PCA'), PipelineElement('FastICA')]),
                                  Branch('MyBranch', [PipelineElement('PCA')])])
        self.assertEqual(len(stack.elements), 4)

        # test doubled item
        with self.assertRaises(ValueError):
            stack += stack.elements[0]

        stack += PipelineElement('PCA', {'n_components': [10, 20]})
        self.assertEqual(stack.elements[-1].name, 'PCA2')
        self.assertDictEqual(stack.hyperparameters, {'MyStack__MySwitch__current_element': [(0, 0), (1, 0)],
                                                     'MyStack__PCA2__n_components': [10, 20]})

    def test_feature_importances(self):
        # single item
        self.estimator_stack.fit(self.X, self.y)
        self.assertIsNone(self.estimator_stack.feature_importances_)

        self.estimator_branch_stack.fit(self.X, self.y)
        self.assertIsNone(self.estimator_branch_stack.feature_importances_)

    def test_use_probabilities(self):
        self.estimator_stack.use_probabilities = True
        self.estimator_stack.fit(self.X, self.y)
        probas = self.estimator_stack.predict(self.X)
        self.assertEqual(probas.shape[1], 3)

        self.estimator_stack.use_probabilities = False
        self.estimator_stack.fit(self.X, self.y)
        preds = self.estimator_stack.predict(self.X)
        self.assertEqual(preds.shape[1], 2)
        probas = self.estimator_stack.predict_proba(self.X)
        self.assertEqual(probas.shape[1], 3)


class PreprocessingTests(unittest.TestCase):

    def test_hyperparameter_add(self):
        pe = Preprocessing()
        with self.assertRaises(ValueError):
            pe.add(PipelineElement('PCA', hyperparameters={'n_components': [1, 5]}))

    def test_predict_warning(self):
        pe = Preprocessing()
        pe.add(PipelineElement('SVC'))
        with warnings.catch_warnings(record=True) as w:
            pe.predict([0, 1, 2])
            assert any("There is no predict function" in s for s in [e.message.args[0] for e in w])


class DataFilterTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.filter_1 = DataFilter(indices=[0, 1, 2, 3, 4])
        self.filter_2 = DataFilter(indices=[5, 6, 7, 8, 9])

    def test_filter(self):
        Xt_1, y_1, _ = self.filter_1.transform(self.X, self.y)
        Xt_2, y_2, _ = self.filter_2.transform(self.X, self.y)

        self.assertTrue(np.array_equal(self.y, y_1))
        self.assertTrue(np.array_equal(self.y, y_2))
        self.assertTrue(np.array_equal(Xt_1, self.X[:, :5]))
        self.assertTrue(np.array_equal(Xt_2, self.X[:, 5:10]))


class CallbackElementTests(unittest.TestCase):

    def setUp(self):
        def callback(X, y=None, **kwargs):
            self.assertEqual(X.shape, (569, 30))
            print("Shape of transformed data: {}".format(X.shape))

        def predict_callback(X, y=None, **kwargs):
            self.assertEqual(X.shape, (569, ))
            print('Shape of predictions: {}'.format(X.shape))

        def callback_test_equality(X, y=None, **kwargs):
            self.assertTrue(np.array_equal(self.X, X))
            if y is not None:
                self.assertListEqual(self.y.tolist(), y.tolist())

        self.X, self.y = load_breast_cancer(return_X_y=True)

        self.clean_pipeline = PhotonPipeline(elements=[('PCA', PipelineElement('PCA')),
                                                       ('LogisticRegression', PipelineElement('LogisticRegression'))])
        self.callback_pipeline = PhotonPipeline(elements=[('First', CallbackElement('First', callback)),
                                                          ('PCA', PipelineElement('PCA')),
                                                          ('Second', CallbackElement('Second', callback)),
                                                          ('LogisticRegression', PipelineElement('LogisticRegression'))])
        self.clean_branch_pipeline = PhotonPipeline(elements=[('MyBranch',
                                                               Branch('MyBranch', [PipelineElement('PCA')])),
                                                              ('LogisticRegression',
                                                               PipelineElement('LogisticRegression'))])
        self.callback_branch_pipeline = PhotonPipeline(elements=[('First', CallbackElement('First', callback)),
                                                                 ('MyBranch', Branch('MyBranch', [CallbackElement('Second',
                                                                                                                  callback),
                                                                                                  PipelineElement('PCA')])),
                                                                 ('Fourth', CallbackElement('Fourth', callback)),
                                                                 ('LogisticRegression',
                                                                  PipelineElement('LogisticRegression'))])
        self.callback_branch_pipeline_error = PhotonPipeline(elements=[('First', CallbackElement('First', callback)),
                                                                       ('MyBranch', Branch('MyBranch', [CallbackElement('Second',
                                                                                                                        callback),
                                                                                                        PipelineElement('PCA'),
                                                                                                        CallbackElement('Third',
                                                                                                                        callback)])),
                                                                       ('Fourth', CallbackElement('Fourth', callback)),
                                                                       ('LogisticRegression',
                                                                        PipelineElement('LogisticRegression')),
                                                                       ('Fifth', CallbackElement('Fifth', predict_callback))])
        # test that data is unaffected from pipeline
        self.callback_after_callback_pipeline = PhotonPipeline([('Callback1', CallbackElement('Callback1', callback)),
                                                                ('Callback2', CallbackElement('Callback2', callback_test_equality)),
                                                                ('StandarcScaler', PipelineElement('StandardScaler'),
                                                                 ('SVR', PipelineElement('SVR')))])

    def test_callback(self):
        pipelines = [self.clean_pipeline, self.callback_pipeline, self.clean_branch_pipeline,
                     self.callback_branch_pipeline, self.callback_after_callback_pipeline]

        for pipeline in pipelines:
            pipeline.fit(self.X, self.y).predict(self.X)

        with warnings.catch_warnings(record=True) as w:
            self.callback_branch_pipeline_error.fit(self.X, self.y).predict(self.X)
            assert any("Last element of pipeline cannot be callback" in s for s in [e.message.args[0] for e in w])

