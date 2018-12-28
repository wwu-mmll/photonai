import unittest

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from photonai.base.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch


class HyperpipeTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.cv_object = KFold(n_splits=3)
        self.hyperpipe = Hyperpipe('god', self.cv_object)
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

    def test_init(self):
        self.assertEqual(self.hyperpipe.name, 'god')
        # assure pipeline has two steps, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe._pipe.steps), 2)
        self.assertIs(self.hyperpipe._pipe.steps[0][1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe._pipe.steps[1][1], self.svc_pipe_element)


class PipelineElementTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})

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


    def test_set_params(self):
        config = {'n_components': 3, 'disabled': False}
        self.pca_pipe_element.set_params(**config)
        self.assertFalse(self.pca_pipe_element.disabled)
        self.assertEqual(self.pca_pipe_element.base_element.n_components, 3)
        with self.assertRaises(ValueError):
            self.pca_pipe_element.set_params(**{'any_weird_param': 1})

    def test_fit(self):
        pass


class PipelineSwitchTests(unittest.TestCase):

    def setUp(self):
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.lr_pipe_element = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]})
        self.pipe_switch = PipelineSwitch('switch', [self.svc_pipe_element, self.lr_pipe_element])

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
        false_config = {'current_element': 1}
        with self.assertRaises(ValueError):
            self.pipe_switch.set_params(**false_config)

        correct_config = {'current_element': (0, 1)}
        self.pipe_switch.set_params(**correct_config)
        self.assertEqual(self.pipe_switch.base_element.base_element.C, 0.1)
        self.assertEqual(self.pipe_switch.base_element.base_element.kernel, 'sigmoid')

    def test_base_element(self):
        self.pipe_switch.set_params(**{'switch__current_element': (1, 1)})
        self.assertIs(self.pipe_switch.base_element, self.lr_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.lr_pipe_element.base_element)


if __name__ == '__main__':
    unittest.main()


