import unittest
import types

from photonai.base import PipelineElement
from photonai.optimization import IntegerRange

try:
    from photonai.optimization.smac.smac3 import SMACOptimizer
    found = True
except ModuleNotFoundError:
    found = False

if found:
    class SMACOptimizerWithRequirementsTest(unittest.TestCase):

        def setUp(self):
            """
            Set up for SmacOptimizer.
            """
            self.pipeline_elements = [PipelineElement("StandardScaler"),
                                      PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                      test_disabled=True),
                                      PipelineElement("SVC")]
            self.smac_optimizer = SMACOptimizer()

        def test_all_attributes_available(self):
            """
            Test for .ask attribute. .ask is important for next configuration that should be tested.
            """
            self.smac_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
            self.assertIsInstance(self.smac_optimizer.ask, types.GeneratorType)

else:
    class SMACOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.smac.smac3 import SMACOptimizer