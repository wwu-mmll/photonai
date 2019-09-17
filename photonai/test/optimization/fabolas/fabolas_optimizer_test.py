import types
import unittest

from photonai.base import PipelineElement
from photonai.optimization import IntegerRange

try:
    from photonai.optimization.fabolas.fabolas_optimizer import FabolasOptimizer

    found = True
except ModuleNotFoundError:
    found = False

if found:
    class FabolasOptimizerWithRequirementsTest(unittest.TestCase):

        def setUp(self):
            """
            Set up for FabolasOptimizer.
            """
            self.pipeline_elements = [PipelineElement("StandardScaler"),
                                      PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                      test_disabled=True),
                                      PipelineElement("SVC")]
            self.fabolas_optimizer = FabolasOptimizer(n_min_train_data=100, n_train_data=10000)

        def test_all_attributes_available(self):
            """
            Test for .ask and .param_grid attribute. .ask is important for next configuration that should be tested.
            """
            self.fabolas_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
            self.assertIsInstance(self.fabolas_optimizer.ask, types.GeneratorType)

else:
    class FabolasOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.fabolas.fabolas_optimizer import FabolasOptimizer
