import unittest
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.datasets import load_breast_cancer

from photonai.base import PipelineElement, Switch, Stack, Branch, Preprocessing
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.test.base.dummy_elements import DummyEstimator, \
    DummyNeedsCovariatesEstimator, DummyNeedsCovariatesTransformer, DummyNeedsYTransformer, DummyTransformer, \
    DummyNeedsCovariatesAndYTransformer


def elements_to_dict(elements):
    if isinstance(elements, dict):
        new_dict = dict()
        for name, element in elements.items():
            new_dict[name] = elements_to_dict(element)
        elements = new_dict
    elif isinstance(elements, list):
        new_list = list()
        for element in elements:
            new_list.append(elements_to_dict(element))
        elements = new_list
    elif isinstance(elements, tuple):
        new_list = list()
        for element in elements:
            new_list.append(elements_to_dict(element))
        elements = tuple(new_list)
    elif isinstance(elements, (Switch, Branch, Preprocessing, Stack, PhotonPipeline)):
        new_dict = dict()
        elements = elements.__dict__
        for name, element in elements.items():
            new_dict[name] = elements_to_dict(element)
        elements = new_dict
    elif isinstance(elements, PipelineElement):
        new_dict = dict()
        elements = elements.__dict__
        if not isinstance(elements["base_element"], dict):
            new_dict["base_element"] = elements["base_element"].__dict__
        elements = new_dict
    return elements


class PipelineElementTests(unittest.TestCase):

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

    def test_copy_me(self):
        svc = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        svc.set_params(**{'C': 0.1, 'kernel': 'sigmoid'})
        copy = svc.copy_me()
        self.assertNotEqual(copy.base_element, svc.base_element)
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(svc))
        self.assertEqual(copy.base_element.C, svc.base_element.C)

        svc = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        copy = svc.copy_me()
        self.assertDictEqual(copy.hyperparameters, {'SVC__C': [0.1, 1], 'SVC__kernel': ['rbf', 'sigmoid']})
        copy.base_element.C = 3
        self.assertNotEqual(svc.base_element.C, copy.base_element.C)


class SwitchTests(unittest.TestCase):

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
        switches = [self.pipe_switch, self.pipe_switch_with_branch, self.pipe_transformer_switch_with_branch,
                    self.switch_in_switch]

        for switch in switches:
            copy = switch.copy_me()

            for i, element in enumerate(copy.elements):
                self.assertNotEqual(copy.elements[i], switch.elements[i])

            switch = elements_to_dict(switch)
            copy = elements_to_dict(copy)

            self.assertDictEqual(copy, switch)


class BranchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)
        self.ss_pipe_element = PipelineElement("StandardScaler", {'with_mean': True})
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True, random_state=3)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']}, random_state=3)
        self.branch = Branch('MyBranch')
        self.branch += self.ss_pipe_element
        self.branch += self.pca_pipe_element

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

    def test_copy_me(self):
        branch = Branch('MyBranch')
        branch += self.ss_pipe_element
        branch += self.pca_pipe_element

        copy = branch.copy_me()
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(branch))

        copy = branch.copy_me()
        copy.elements[1].base_element.n_components = 3
        self.assertNotEqual(copy.elements[1].base_element.n_components, branch.elements[1].base_element.n_components)

        fake_copy = branch
        fake_copy.elements[1].base_element.n_components = 3
        self.assertEqual(fake_copy.elements[1].base_element.n_components, branch.elements[1].base_element.n_components)

    def test_prepare_pipeline(self):
        self.assertEqual(len(self.branch.elements), 2)
        config_grid = {'MyBranch__PCA__n_components': [1, 2],
                       'MyBranch__PCA__disabled': [False, True],
                       'MyBranch__StandardScaler__with_mean': True}
        self.assertDictEqual(config_grid, self.branch._hyperparameters)


class StackTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)

        self.trans_1 = PipelineElement('PCA')
        self.trans_2 = PipelineElement('StandardScaler')
        self.est_1 = PipelineElement('SVC')
        self.est_2 = PipelineElement('DecisionTreeClassifier')

        self.transformer_branch_1 = Branch('TransBranch1')
        self.transformer_branch_1 += self.trans_1
        self.transformer_branch_2 = Branch('TransBranch2')
        self.transformer_branch_2 += self.trans_2

        self.estimator_branch_1 = Branch('EstBranch1')
        self.estimator_branch_1 += self.est_1
        self.estimator_branch_2 = Branch('EstBranch2')
        self.estimator_branch_2 += self.est_2

        self.transformer_stack = Stack('TransformerStack', [self.trans_1.copy_me(), self.trans_2.copy_me()])
        self.estimator_stack = Stack('EstimatorStack', [self.est_1.copy_me(), self.est_2.copy_me()])
        self.transformer_branch_stack = Stack('TransBranchStack', [self.transformer_branch_1.copy_me(),
                                                                   self.transformer_branch_2.copy_me()])
        self.estimator_branch_stack = Stack('EstBranchStack', [self.estimator_branch_1.copy_me(),
                                                               self.estimator_branch_2.copy_me()])

        self.stacks = [([self.trans_1, self.trans_2], self.transformer_stack),
                       ([self.est_1, self.est_2], self.estimator_stack),
                       ([self.transformer_branch_1, self.transformer_branch_2], self.transformer_branch_stack),
                       ([self.estimator_branch_1, self.estimator_branch_2], self.estimator_branch_stack)]

    def test_copy_me(self):
        for stack in self.stacks:
            stack = stack[1]
            copy = stack.copy_me()
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
