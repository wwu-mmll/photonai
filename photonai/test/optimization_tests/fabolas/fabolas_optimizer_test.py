import unittest

from photonai.base import PipelineElement
from photonai.optimization import IntegerRange
from photonai.test.optimization.grid_search.grid_search_test import GridSearchOptimizerTest

try:
    from photonai.optimization.fabolas.fabolas_optimizer import FabolasOptimizer

    found = True
except ModuleNotFoundError:
    found = False

if found:
    class FabolasOptimizerWithRequirementsTest(GridSearchOptimizerTest):

        def setUp(self):
            """
            Set up for FabolasOptimizer.
            """
            self.pipeline_elements = [PipelineElement("StandardScaler"),
                                      PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                      test_disabled=True),
                                      PipelineElement("SVC")]
            self.optimizer = FabolasOptimizer(n_min_train_data=100, n_train_data=10000)

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
    class FabolasOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.fabolas.fabolas_optimizer import FabolasOptimizer
