import unittest
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.datasets import load_breast_cancer

from photonai.base import PipelineElement, Switch, Stack, Branch, Preprocessing, DataFilter, CallbackElement
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.test.base.dummy_elements import DummyEstimator, \
    DummyNeedsCovariatesEstimator, DummyNeedsCovariatesTransformer, DummyNeedsYTransformer, DummyTransformer, \
    DummyNeedsCovariatesAndYTransformer, DummyEstimatorNoPredict, DummyEstimatorWrongType, DummyTransformerWithPredict


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

    def test_fit(self):
        self.pca_pipe_element.fit(self.X, self.y)
        self.assertEqual(self.pca_pipe_element.base_element.components_.shape, (30, 30))
        self.assertEqual(self.pca_pipe_element.base_element.components_[0, 0], 0.005086232018734175)

        self.svc_pipe_element.fit(self.X, self.y)
        self.assertEqual(self.svc_pipe_element.base_element._intercept_, -0.3753900173819406)

    def test_transform(self):
        self.pca_pipe_element.fit(self.X, self.y)

        Xt, _, _ = self.pca_pipe_element.transform(self.X)
        self.assertEqual(Xt.shape, (569, 30))
        self.assertEqual(Xt[0, 0], 1160.1425737041347)

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

        # test custom element
        custom_element = PipelineElement.create('CustomElement', base_element=DummyNeedsCovariatesEstimator(),
                                                hyperparameters={})
        copy = custom_element.copy_me()
        self.assertDictEqual(elements_to_dict(custom_element), elements_to_dict(copy))

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


class SwitchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.tree_pipe_element = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]})
        self.pipe_switch = Switch('switch', [self.svc_pipe_element.copy_me(), self.tree_pipe_element.copy_me()])
        self.branch = Branch('branch')
        self.branch += self.svc_pipe_element
        self.transformer_branch = Branch('transformer_branch')
        self.transformer_branch += PipelineElement('PCA')
        self.transformer = PipelineElement('PCA')
        self.pipe_switch_with_branch = Switch('switch_with_branch', [self.tree_pipe_element, self.branch])
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

    def test_fit(self):
        self.pipe_switch.set_params(**{'current_element': (1, 1)})
        self.pipe_switch.fit(self.X, self.y)
        self.tree_pipe_element.fit(self.X, self.y)
        np.testing.assert_array_equal(self.tree_pipe_element.base_element.feature_importances_,
                                      self.pipe_switch.base_element.feature_importances_)

    def test_transform(self):
        self.pipe_switch.set_params(**{'current_element': (1, 1)})
        self.pipe_switch.fit(self.X, self.y)
        self.tree_pipe_element.fit(self.X, self.y)

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_inverse_transform(self):
        pass

    def test_base_element(self):
        # grid search
        self.pipe_switch.set_params(**{'current_element': (1, 1)})
        self.assertIs(self.pipe_switch.base_element, self.tree_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.tree_pipe_element.base_element)

        # other optimizer
        self.pipe_switch.set_params(**{'DecisionTreeClassifier__min_samples_split': 2})
        self.assertIs(self.pipe_switch.base_element, self.tree_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.tree_pipe_element.base_element)

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

        self.assertEqual(self.pipe_switch._estimator_type, 'classifier')
        self.assertEqual(self.pipe_switch_with_branch._estimator_type, 'classifier')
        self.assertEqual(self.pipe_transformer_switch_with_branch._estimator_type, None)
        self.assertEqual(self.switch_in_switch._estimator_type, None)

    def test_add(self):
        self.assertEqual(len(self.pipe_switch.elements), 2)
        self.assertEqual(len(self.switch_in_switch.elements), 2)
        self.assertEqual(len(self.pipe_transformer_switch_with_branch.elements), 2)

        self.assertEqual(list(self.pipe_switch.elements_dict.keys()), ['SVC', 'DecisionTreeClassifier'])
        self.assertEqual(list(self.switch_in_switch.elements_dict.keys()), ['transformer_branch',
                                                                      'transformer_switch_with_branch'])

        switch = Switch('MySwitch', [PipelineElement('PCA'), PipelineElement('FastICA')])
        switch = Switch('MySwitch')
        switch += PipelineElement('PCA')
        switch += PipelineElement('FastICA')

        with self.assertRaises(Exception):
            self.pipe_switch += self.pipe_switch.elements[0]


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

    def test_fit(self):
        pass

    def test_transform(self):
        pass

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_inverse_transform(self):
        pass

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

    def test_set_params(self):
        config = {'PCA__n_components': 2,
                  'PCA__disabled': True,
                  'StandardScaler__with_mean': True}
        self.branch.set_params(**config)
        self.assertTrue(self.branch.base_element.elements[1][1].disabled)
        self.assertEqual(self.branch.base_element.elements[1][1].base_element.n_components, 2)
        self.assertEqual(self.branch.base_element.elements[0][1].base_element.with_mean, True)

        with self.assertRaises(ValueError):
            self.branch.set_params(**{'any_weird_param': 1})

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


class StackTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)

        self.trans_1 = PipelineElement('PCA', {'n_components': [5, 10]})
        self.trans_2 = PipelineElement('StandardScaler', {'with_mean': [True]})
        self.est_1 = PipelineElement('SVC', {'C': [1, 2]})
        self.est_2 = PipelineElement('DecisionTreeClassifier', {'min_samples_leaf': [3, 5]})

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

    def test_fit(self):
        pass

    def test_transform(self):
        pass

    def test_predict(self):
        pass

    def test_predict_proba(self):
        pass

    def test_inverse_transform(self):
        pass

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


class DataFilterTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)
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

        self.X, self.y = load_breast_cancer(True)

        self.clean_pipeline = PhotonPipeline(elements=[('PCA', PipelineElement('PCA')),
                                                       ('LogisticRegression', PipelineElement('LogisticRegression'))])
        self.callback_pipeline = PhotonPipeline(elements=[('First', CallbackElement('First', callback)),
                                                          ('PCA', PipelineElement('PCA')),
                                                          ('Second', CallbackElement('Second', callback)),
                                                          ('LogisticRegression', PipelineElement('LogisticRegression')),
                                                          ('Third', CallbackElement('Third', predict_callback))])
        self.clean_branch_pipeline = PhotonPipeline(elements=[('MyBranch',
                                                               Branch('MyBranch', [PipelineElement('PCA')])),
                                                              ('LogisticRegression',
                                                               PipelineElement('LogisticRegression'))])
        self.callback_branch_pipeline = PhotonPipeline(elements=[('First', CallbackElement('First', callback)),
                                                                 ('MyBranch', Branch('MyBranch', [CallbackElement('Second',
                                                                                                     callback),
                                                                                     PipelineElement('PCA'),
                                                                                     CallbackElement('Third',
                                                                                                     callback)])),
                                                                 ('Fourth', CallbackElement('Fourth', callback)),
                                                                 ('LogisticRegression',
                                                                  PipelineElement('LogisticRegression')),
                                                                 ('Fifth', CallbackElement('Fifth', predict_callback))])

    def test_callback(self):
        pipelines = [self.clean_pipeline, self.callback_pipeline, self.clean_branch_pipeline,
                     self.callback_branch_pipeline]

        for pipeline in pipelines:
            pipeline.fit(self.X, self.y).predict(self.X)
