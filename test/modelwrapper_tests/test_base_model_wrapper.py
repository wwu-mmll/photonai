import unittest

from photonai.modelwrapper.base_model_wrapper import BaseModelWrapper

class BaseModelWrapperTest(unittest.TestCase):
    """
    Simple testclass for abstract BaseModelWrapper.
    Check required functions 'fit' and 'transform'.
    """

    def setUp(self):
        self.model_wrapper = BaseModelWrapper()

    def test_methods_available(self):
        if hasattr(self.model_wrapper, '_estimator_type'):
            if self.model_wrapper._estimator_type in ['classifiern', 'regressor']:
                methods = ['fit', 'predict']
                for method in methods:
                    self.assertTrue(
                        (hasattr(self.model_wrapper, method) and callable(getattr(self.model_wrapper, method))))
            elif self.model_wrapper._estimator_type == 'transformer':
                methods = ['fit', 'transform']
                for method in methods:
                    self.assertTrue(
                        (hasattr(self.model_wrapper, method) and callable(getattr(self.model_wrapper, method))))