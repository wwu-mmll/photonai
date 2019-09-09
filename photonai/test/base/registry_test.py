import unittest
import os
import numpy as np

from sklearn.model_selection import KFold

from photonai.base import PipelineElement, Hyperpipe, OutputSettings
from photonai.base import PhotonRegistry


class RegistryTest(unittest.TestCase):

    def setUp(self):
        self.custom_folder = "./custom_elements"
        self.registry = PhotonRegistry()

    def test_custom_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.registry.activate("./some_not_existing_folder")

    def test_python_file_not_in_custom_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.registry.register('SomeName', 'not_existing_file.SomeName', 'Estimator', './custom_elements')

    def test_list_available_elements(self):
        self.registry.list_available_elements()
        self.registry.list_available_elements('PhotonNeuro')
        self.registry.info('PCA')
        self.registry.info('NotExistingEstimator')

    def test_register_element(self):
        with self.assertRaises(ValueError):
            self.registry.register('MyCustomEstimator', 'custom_estimator.CustomEstimator', 'WrongType', './custom_elements')

        self.registry.register('MyCustomEstimator', 'custom_estimator.CustomEstimator', 'Estimator', './custom_elements')

        self.registry.activate(self.custom_folder)
        settings = OutputSettings(save_output=False)

        # DESIGN YOUR PIPELINE
        pipe = Hyperpipe('custom_estimator_pipe',
                         optimizer='random_grid_search',
                         optimizer_params={'k': 2},
                         metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                         best_config_metric='accuracy',
                         outer_cv=KFold(n_splits=2),
                         inner_cv=KFold(n_splits=2),
                         verbosity=1,
                         output_settings=settings)

        pipe += PipelineElement('MyCustomEstimator')

        pipe.fit(np.random.randn(30, 30), np.random.randint(0, 2, 30))

        self.registry.delete('MyCustomEstimator')

        os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))

    def test_estimator_check_during_register(self):
        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoFit', 'custom_estimator.CustomEstimatorNoFit', 'Estimator',
                                   './custom_elements')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoPredict', 'custom_estimator.CustomEstimatorNoPredict',
                                   'Estimator', './custom_elements')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoEstimatorType', 'custom_estimator.CustomEstimatorNoEstimatorType',
                                   'Estimator', './custom_elements')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNotReturningSelf',
                                       'custom_estimator.CustomEstimatorNotReturningSelf',
                                       'Estimator', './custom_elements')

        e = self.registry.register('MyCustomEstimatorReturningFalsePredictions',
                                   'custom_estimator.CustomEstimatorReturningFalsePredictions',
                                   'Estimator', './custom_elements')
        self.assertIsInstance(e, ValueError)

        e = self.registry.register('MyCustomEstimatorNotWorking', 'custom_estimator.CustomEstimatorNotWorking',
                                   'Estimator', './custom_elements')
        self.assertIsInstance(e, ValueError)

        os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))

    def test_transformer_needs_covariates(self):
        self.registry.register('MyCustomTransformerNeedsCovariates', 'custom_transformer.CustomTransformerNeedsCovariates',
                               'Transformer', './custom_elements')

        with self.assertRaises(ValueError):
            self.registry.register('MyCustomTransformerNeedsCovariatesWrongInterface',
                                   'custom_transformer.CustomTransformerNeedsCovariatesWrongInterface',
                                   'Transformer', './custom_elements')

    def test_transformer_needs_y(self):
        self.registry.register('MyCustomTransformerNeedsY',
                               'custom_transformer.CustomTransformerNeedsY',
                               'Transformer', './custom_elements')

        with self.assertRaises(ValueError):
            self.registry.register('MyCustomTransformerNeedsYWrongInterface',
                                   'custom_transformer.CustomTransformerNeedsYWrongInterface',
                                   'Transformer', './custom_elements')

    def tearDown(self):
        if os.path.isfile(os.path.join(self.custom_folder, 'CustomElements.json')):
            os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))


