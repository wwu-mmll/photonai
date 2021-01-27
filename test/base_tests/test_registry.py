import os

import numpy as np
from sklearn.model_selection import KFold

from photonai.base import PhotonRegistry
from photonai.base import PipelineElement, Hyperpipe, OutputSettings
from photonai.helper.photon_base_test import PhotonBaseTest


class RegistryTest(PhotonBaseTest):


    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(RegistryTest, cls).setUpClass()

    def setUp(self):
        super(RegistryTest, self).setUp()
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('test')[0], 'test')
        self.custom_folder = os.path.join(self.test_directory, 'base_tests/custom_elements/')
        self.registry = PhotonRegistry(self.custom_folder)

    def tearDown(self):
        super(RegistryTest, self).tearDown()
        if os.path.isfile(os.path.join(self.custom_folder, 'CustomElements.json')):
            os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))

    def test_register_without_custom_folder(self):
        registry = PhotonRegistry()
        with self.assertRaises(ValueError):
            registry.register('SomeName', 'not_existing_file.SomeName', 'Estimator')

    def test_python_file_not_in_custom_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.registry.register('SomeName', 'not_existing_file.SomeName', 'Estimator')

    def test_list_available_elements(self):
        self.registry.list_available_elements()
        self.registry.info('PCA')
        self.registry.info('NotExistingEstimator')

    def test_register_element(self):
        with self.assertRaises(ValueError):
            self.registry.register('MyCustomEstimator', 'custom_estimator.CustomEstimator', 'WrongType')

        self.registry.register('MyCustomEstimator', 'custom_estimator.CustomEstimator', 'Estimator')

        self.registry.activate()
        settings = OutputSettings(save_output=False)

        # DESIGN YOUR PIPELINE
        pipe = Hyperpipe('custom_estimator_pipe',
                         optimizer='random_grid_search',
                         optimizer_params={'n_configurations': 2},
                         metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                         best_config_metric='accuracy',
                         outer_cv=KFold(n_splits=2),
                         inner_cv=KFold(n_splits=2),
                         verbosity=1,
                         project_folder = './tmp/',
                         output_settings=settings)

        pipe += PipelineElement('MyCustomEstimator')

        pipe.fit(np.random.randn(30, 30), np.random.randint(0, 2, 30))

        self.registry.delete('MyCustomEstimator')

        os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))

    def test_estimator_check_during_register(self):
        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoFit', 'custom_estimator.CustomEstimatorNoFit', 'Estimator')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoPredict', 'custom_estimator.CustomEstimatorNoPredict',
                                   'Estimator')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNoEstimatorType',
                                   'custom_estimator.CustomEstimatorNoEstimatorType',
                                   'Estimator')

        with self.assertRaises(NotImplementedError):
            self.registry.register('MyCustomEstimatorNotReturningSelf',
                                   'custom_estimator.CustomEstimatorNotReturningSelf',
                                   'Estimator')

        e = self.registry.register('MyCustomEstimatorReturningFalsePredictions',
                                   'custom_estimator.CustomEstimatorReturningFalsePredictions',
                                   'Estimator')
        self.assertIsInstance(e, ValueError)

        e = self.registry.register('MyCustomEstimatorNotWorking', 'custom_estimator.CustomEstimatorNotWorking',
                                   'Estimator')

        self.assertIsInstance(e, ValueError)

        os.remove(os.path.join(self.custom_folder, 'CustomElements.json'))

    def test_transformer_needs_covariates(self):
        self.registry.register('MyCustomTransformerNeedsCovariates',
                               'custom_transformer.CustomTransformerNeedsCovariates',
                               'Transformer')
        with self.assertRaises(ValueError):
            self.registry.register('MyCustomTransformerNeedsCovariatesWrongInterface',
                                   'custom_transformer.CustomTransformerNeedsCovariatesWrongInterface',
                                   'Transformer')

    def test_transformer_needs_y(self):
        self.registry.register('MyCustomTransformerNeedsY',
                               'custom_transformer.CustomTransformerNeedsY',
                               'Transformer')

        with self.assertRaises(ValueError):
            self.registry.register('MyCustomTransformerNeedsYWrongInterface',
                                   'custom_transformer.CustomTransformerNeedsYWrongInterface',
                                   'Transformer')

    def test_add_module(self):
        json_module_file = os.path.join(self.custom_folder, "fake_module.json")

        # first add module
        self.registry.add_module(json_module_file)
        self.assertTrue("fake_module" in self.registry.PHOTON_REGISTRIES)

        # check if registered items are available
        self.assertTrue("FakeElement1" in self.registry.ELEMENT_DICTIONARY)
        fe = PipelineElement("FakeElement1")

        # then delete the module
        self.registry.delete_module("fake_module")
        self.assertFalse(os.path.isfile(os.path.join(self.registry.module_path, "fake_module.json")))

        self.assertFalse("FakeElement1" in self.registry.ELEMENT_DICTIONARY)
        with self.assertRaises(NameError):
            fe = PipelineElement("FakeElement1")

    def test_add_failing_module(self):
        json_module_file = os.path.join(self.custom_folder, "not_working_fake_module.json")

        # try to add module
        self.registry.add_module(json_module_file)

        # that should not have worked because we cant import first element
        self.assertFalse("FakeElement2001" in self.registry.ELEMENT_DICTIONARY)
        self.assertFalse(os.path.isfile(os.path.join(self.registry.module_path, "not_working_fake_module.json")))




