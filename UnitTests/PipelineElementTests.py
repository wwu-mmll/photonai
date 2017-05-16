import unittest
import numpy as np

from HPOFramework.HPOBaseClasses import PipelineElement, Hyperpipe, PipelineSwitch, PipelineFusion
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

class HyperpipeTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement.create('pca', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement.create('svc', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.cv_object = KFold(n_splits=3)
        self.hyperpipe = Hyperpipe('god', self.cv_object)
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

    def test_init(self):
        self.assertEqual(self.hyperpipe.name, 'god')
        # assure pipeline has two steps, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe.pipe.steps), 2)
        self.assertIs(self.hyperpipe.pipe.steps[0][1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe.pipe.steps[1][1], self.svc_pipe_element)

    def test_hyperparameters(self):
        # hyperparameters
        self.assertDictEqual(self.hyperpipe.hyperparameters, {'pca': {'n_components': [1, 2], 'set_disabled': True},
                                                              'svc': {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']}})
        # sklearn params
        # Todo: has no sklearn attribute
        # config grid
        print(self.hyperpipe.config_grid)
        expected_config_grid = [{'pca__n_components': 1, 'pca__disabled': False, 'svc__C': 0.1, 'svc__kernel': 'rbf'},
                                {'pca__n_components': 1, 'pca__disabled': False, 'svc__C': 0.1, 'svc__kernel': 'sigmoid'},
                                {'pca__n_components': 1, 'pca__disabled': False, 'svc__C': 1, 'svc__kernel': 'rbf'},
                                {'pca__n_components': 1, 'pca__disabled': False, 'svc__C': 1, 'svc__kernel': 'sigmoid'},
                                {'pca__n_components': 2, 'pca__disabled': False, 'svc__C': 0.1, 'svc__kernel': 'rbf'},
                                {'pca__n_components': 2, 'pca__disabled': False, 'svc__C': 0.1, 'svc__kernel': 'sigmoid'},
                                {'pca__n_components': 2, 'pca__disabled': False, 'svc__C': 1, 'svc__kernel': 'rbf'},
                                {'pca__n_components': 2, 'pca__disabled': False, 'svc__C': 1,'svc__kernel': 'sigmoid'},
                                {'pca__disabled': True, 'svc__C': 0.1, 'svc__kernel': 'rbf'},
                                {'pca__disabled': True, 'svc__C': 0.1, 'svc__kernel': 'sigmoid'},
                                {'pca__disabled': True, 'svc__C': 1, 'svc__kernel': 'rbf'},
                                {'pca__disabled': True, 'svc__C': 1, 'svc__kernel': 'sigmoid'}]
        expected_config_grid = [sorted(i) for i in expected_config_grid]
        actual_config_grid = [sorted(i) for i in self.hyperpipe.config_grid]
        self.assertListEqual(actual_config_grid, expected_config_grid)


class PipelineElementTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement.create('pca', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement.create('svc', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})

    def tearDown(self):
        pass

    def test_create_failure(self):
        with self.assertRaises(NameError):
            PipelineElement.create('dusihdaushdisuhdusiahd', {})

    def test_pipeline_element_create(self):
        # test name, set_disabled and base_element
        self.assertIsInstance(self.pca_pipe_element.base_element, PCA)

        # set_disabled is passed correctly
        self.assertTrue(self.pca_pipe_element.set_disabled)
        # correct name
        self.assertEqual(self.pca_pipe_element.name, 'pca')

    def test_one_hyperparameter_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.pca_pipe_element.sklearn_hyperparams, {'pca__n_components': [1, 2]})
        # config_grid is created as expected
        self.assertListEqual(self.pca_pipe_element.config_grid, [{'pca__n_components': 1, 'pca__disabled': False},
                                                                 {'pca__n_components': 2, 'pca__disabled': False},
                                                                 {'pca__disabled': True}])
        # hyperparameter dictionary is returned as expected
        self.assertDictEqual(self.pca_pipe_element.hyperparameters, {'n_components': [1, 2], 'set_disabled': True})

    def test_more_hyperparameters_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.svc_pipe_element.sklearn_hyperparams, {'svc__C': [0.1, 1],
                                                                         'svc__kernel': ['rbf', 'sigmoid']})
        # config_grid is created as expected
        self.assertListEqual(self.svc_pipe_element.config_grid, [{'svc__C': 0.1, 'svc__kernel': 'rbf'},
                                                                 {'svc__C': 0.1, 'svc__kernel': 'sigmoid'},
                                                                 {'svc__C': 1, 'svc__kernel': 'rbf'},
                                                                 {'svc__C': 1, 'svc__kernel': 'sigmoid'}])
        # hyperparameter dictionary is returned as expected
        self.assertDictEqual(self.svc_pipe_element.hyperparameters, {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})

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
        self.svc_pipe_element = PipelineElement.create('svc', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.lr_pipe_element = PipelineElement.create('logistic', {'C': [0.1, 0.3, 1]})
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
        self.assertDictEqual(self.pipe_switch.hyperparameters, {'current_element': [(0, 0), (0, 1), (0, 2), (0, 3),
                                                                                    (1, 0), (1, 1), (1, 2)]})

        # sklearn dict
        self.assertDictEqual(self.pipe_switch.sklearn_hyperparams, {'switch__current_element': [(0, 0), (0, 1), (0, 2),
                                                                                                (0, 3), (1, 0), (1, 1),
                                                                                                (1, 2)]})

        # config grid
        self.assertListEqual(self.pipe_switch.config_grid, [{'switch__current_element': (0, 0)},
                                                            {'switch__current_element': (0, 1)},
                                                            {'switch__current_element': (0, 2)},
                                                            {'switch__current_element': (0, 3)},
                                                            {'switch__current_element': (1, 0)},
                                                            {'switch__current_element': (1, 1)},
                                                            {'switch__current_element': (1, 2)}])

        # correct stacking of options
        self.assertDictEqual(self.pipe_switch.pipeline_element_configurations[0][1],
                             self.svc_pipe_element.config_grid[1])

    def test_set_params(self):
        false_config = {'current_element': 1}
        with self.assertRaises(ValueError):
            self.pipe_switch.set_params(**false_config)

        correct_config = {'current_element': (0, 1)}
        self.pipe_switch.set_params(**correct_config)
        self.assertEqual(self.pipe_switch.base_element.base_element.C, 0.1)
        self.assertEqual(self.pipe_switch.base_element.base_element.kernel, 'sigmoid')

    def test_base_element(self):
        self.pipe_switch.set_params(**{'current_element': (1, 1)})
        self.assertIs(self.pipe_switch.base_element, self.lr_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.lr_pipe_element.base_element)


if __name__ == '__main__':
    unittest.main()


