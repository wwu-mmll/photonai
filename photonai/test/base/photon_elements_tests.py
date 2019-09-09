import unittest
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.datasets import load_breast_cancer

from photonai.base import PipelineElement, Switch, Stack, Branch
from photonai.test.base.dummy_elements import DummyEstimator, \
    DummyNeedsCovariatesEstimator, DummyNeedsCovariatesTransformer, DummyNeedsYTransformer, DummyTransformer, \
    DummyNeedsCovariatesAndYTransformer


class PhotonElementsTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.X, self.y = load_breast_cancer(True)
        self.kwargs = {'covariates': self.y}
        self.Xt = self.X + 1
        self.yt = self.y + 1
        self.kwargst = {'covariates': self.y + 1}

    def tearDown(self):
        pass

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


class PipelineSwitchTests(unittest.TestCase):

    def setUp(self):
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.lr_pipe_element = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]})
        self.pipe_switch = Switch('switch', [self.svc_pipe_element, self.lr_pipe_element])
        self.branch = Branch('branch')
        self.branch += self.svc_pipe_element
        self.transformer_branch = Branch('transformer_branch')
        self.transformer_branch += PipelineElement('PCA')
        self.transformer = PipelineElement('PCA')
        self.pipe_switch_with_branch = Switch('switch_with_branch', [self.lr_pipe_element, self.branch])
        self.pipe_transformer_switch_with_branch = Switch('transformer_switch_with_branch',
                                                          [self.transformer, self.transformer_branch])
        self.switch_in_switch = Switch('Switch_in_switch', [self.transformer_branch,
                                                            self.pipe_transformer_switch_with_branch])

    def test_init(self):
        self.assertEqual(self.pipe_switch.name, 'switch')

    def test_hyperparams(self):
        # assert number of different configs to test
        # each config combi for each element: 4 for SVC and 3 for logistic regression = 7
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations), 2)
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations[0]), 4)
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations[1]), 3)

        # hyperparameters
        self.assertDictEqual(self.pipe_switch.hyperparameters, {'switch__current_element': [(0, 0), (0, 1),
                                                                                            (0, 2), (0, 3),
                                                                                            (1, 0), (1, 1),
                                                                                            (1, 2)]})

        # config grid
        self.assertListEqual(self.pipe_switch.generate_config_grid(), [{'switch__current_element': (0, 0)},
                                                                       {'switch__current_element': (0, 1)},
                                                                       {'switch__current_element': (0, 2)},
                                                                       {'switch__current_element': (0, 3)},
                                                                       {'switch__current_element': (1, 0)},
                                                                       {'switch__current_element': (1, 1)},
                                                                       {'switch__current_element': (1, 2)}])

    def test_set_params(self):

        # test for grid search
        false_config = {'current_element': 1}
        with self.assertRaises(ValueError):
            self.pipe_switch.set_params(**false_config)

        correct_config = {'current_element': (0, 1)}
        self.pipe_switch.set_params(**correct_config)
        self.assertEqual(self.pipe_switch.base_element.base_element.C, 0.1)
        self.assertEqual(self.pipe_switch.base_element.base_element.kernel, 'sigmoid')

        # test for other optimizers
        smac_config = {'SVC__C': 2, 'SVC__kernel': 'rbf'}
        self.pipe_switch.set_params(**smac_config)
        self.assertEqual(self.pipe_switch.base_element.base_element.C, 2)
        self.assertEqual(self.pipe_switch.base_element.base_element.kernel, 'rbf')

    def test_base_element(self):
        # grid search
        self.pipe_switch.set_params(**{'current_element': (1, 1)})
        self.assertIs(self.pipe_switch.base_element, self.lr_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.lr_pipe_element.base_element)

        # other optimizer
        self.pipe_switch.set_params(**{'DecisionTreeClassifier__min_samples_split': 2})
        self.assertIs(self.pipe_switch.base_element, self.lr_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.lr_pipe_element.base_element)

    def test_estimator_transformer_check(self):
        self.assertEqual(self.pipe_switch.is_estimator, True)
        self.assertEqual(self.pipe_switch_with_branch.is_estimator, True)
        self.assertEqual(self.pipe_transformer_switch_with_branch.is_estimator, False)
        self.assertEqual(self.switch_in_switch.is_estimator, False)

    def test_copy_me(self):
        # todo
        copy = self.pipe_switch.copy_me()
        self.maxDiff = None
        info_original = self.pipe_switch.__dict__
        info_copy = copy.__dict__

        updated_element_dict = dict()
        for name, element in info_original['elements_dict'].items():
            updated_element_dict[name] = element.__dict__

        updated_elements = list()
        for element in info_original['elements']:
            updated_elements.append(element.__dict__)

        info_original['elements_dict'] = updated_element_dict
        info_original['elements'] = updated_elements

        updated_element_dict = dict()
        for name, element in info_copy['elements_dict'].items():
            updated_element_dict[name] = element.__dict__

        updated_elements = list()
        for element in info_copy['elements']:
            updated_elements.append(element.__dict__)

        info_copy['elements_dict'] = updated_element_dict
        info_copy['elements'] = updated_elements

        self.assertDictEqual(info_copy, info_original)


class PipelineBranchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)
        self.ss_pipe_element = PipelineElement("StandardScaler")
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True, random_state=3)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']}, random_state=3)

    def test_easy_use_case(self):
        sk_pipe = SKPipeline([("SS", StandardScaler()), ("PCA", PCA(random_state=3)), ("SVC", SVC(random_state=3))])
        sk_pipe.fit(self.X, self.y)
        sk_pred = sk_pipe.predict(self.X)

        branch = Branch("my_amazing_branch")
        branch += self.ss_pipe_element
        branch += self.pca_pipe_element
        branch += self.svc_pipe_element
        branch.fit(self.X, self.y)
        branch_pred = branch.predict(self.X)

        self.assertTrue(np.array_equal(sk_pred, branch_pred))

    def test_no_y_transformers(self):
        stacking_element = Stack("forbidden_stack")
        my_dummy = PipelineElement.create("dummy", DummyNeedsCovariatesAndYTransformer(), {})

        with self.assertRaises(NotImplementedError):
            stacking_element += my_dummy

    def test_stacking_of_branches(self):
        branch1 = Branch("B1")
        branch1.add(PipelineElement("StandardScaler"))

        branch2 = Branch("B2")
        branch2.add(PipelineElement("PCA", random_state=3))

        stacking_element = Stack("Stack")
        stacking_element += branch1
        stacking_element += branch2

        stacking_element.fit(self.X, self.y)
        trans, _, _ = stacking_element.transform(self.X)
        pred, _ = stacking_element.predict(self.X)

        self.assertTrue(np.array_equal(trans, pred))
        ss = StandardScaler()
        pca = PCA(random_state=3)
        ss.fit(self.X, self.y)
        X_ss = ss.transform(self.X)
        pca.fit(self.X, self.y)
        X_pca = pca.transform(self.X)

        matrix = np.concatenate((X_ss, X_pca), axis=1)

        self.assertTrue(np.array_equal(trans, matrix))

    def test_voting(self):

        svc1 = PipelineElement("SVC", random_state=1)
        svc2 = PipelineElement("SVC", random_state=1)
        stack_obj = Stack("StackItem", voting=True)
        stack_obj += svc1
        stack_obj += svc2

        sk_svc1 = SVC()
        sk_svc2 = SVC()
        pass
