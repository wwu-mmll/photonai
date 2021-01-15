import unittest

import photonai.optimization.smac.smac as photonai_smac
from photonai.optimization import SMACOptimizer


class SmacOptimizerWithoutRequirementsTest(unittest.TestCase):

    def setUp(self) -> None:
        self._found = photonai_smac.__found__
        photonai_smac.__found__ = False

    def test_imports(self):
        """
        Test for ModuleNotFoundError (requirements.txt).
        """
        with self.assertRaises(ModuleNotFoundError):
            SMACOptimizer()

    def tearDown(self):
        photonai_smac.__found__ = self._found