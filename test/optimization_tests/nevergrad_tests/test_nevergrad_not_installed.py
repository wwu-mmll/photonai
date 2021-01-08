import unittest

import photonai.optimization.nevergrad.nevergrad as photonai_ng
from photonai.optimization import NevergradOptimizer


class NevergradOptimizerWithoutRequirementsTest(unittest.TestCase):

    def setUp(self) -> None:
        self._found = photonai_ng.__found__
        photonai_ng.__found__ = False

    def test_imports(self):
        """
        Test for ModuleNotFoundError (requirements.txt).
        """
        with self.assertRaises(ModuleNotFoundError):
            NevergradOptimizer()

    def tearDown(self):
        photonai_ng.__found__ = self._found
