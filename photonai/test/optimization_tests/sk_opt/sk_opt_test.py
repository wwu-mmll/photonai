from photonai.base import PipelineElement
from photonai.optimization import SkOptOptimizer, IntegerRange
from photonai.test.optimization_tests.grid_search.grid_search_test import GridSearchOptimizerTest


class SkOptOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        """
        Set up for SkOptOptimizerTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                  test_disabled=True),
                                  PipelineElement("SVC")]
        self.optimizer = SkOptOptimizer()

    def test_ask_advanced(self):
        with self.assertRaises(ValueError):
            super(SkOptOptimizerTest, self).test_ask_advanced()
