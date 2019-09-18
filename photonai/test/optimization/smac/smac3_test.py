import unittest

from photonai.base import PipelineElement
from photonai.optimization import IntegerRange
from photonai.test.optimization.grid_search.grid_search_test import GridSearchOptimizerTest

try:
    from photonai.optimization.smac.smac3 import SMACOptimizer

    found = True
except ModuleNotFoundError:
    found = False

if found:
    class SMACOptimizerWithRequirementsTest(GridSearchOptimizerTest):

        def setUp(self):
            """
            Set up for SmacOptimizer.
            """
            self.pipeline_elements = [PipelineElement("StandardScaler"),
                                      PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                      test_disabled=True),
                                      PipelineElement("SVC")]
            self.optimizer = SMACOptimizer()

        def test_all_functions_available(self):
            """
            Test existence of functions and parameters ->  .ask() .tell() .prepare()
            Has to pass cause tell() got additional attribute runtime.
            """
            pass

        def test_ask(self):
            """
            Test general functionality of .ask(). Not implemented yet.
            """
            pass

        def test_ask_advanced(self):
            """
            Test advanced functionality of .ask(). Not implemented yet.
            """
            pass

else:
    class SMACOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.smac.smac3 import SMACOptimizer
