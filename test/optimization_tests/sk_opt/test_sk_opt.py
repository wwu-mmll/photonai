from photonai.base import PipelineElement
from photonai.optimization import SkOptOptimizer, IntegerRange
from ..grid_search.test_grid_search import GridSearchOptimizerTest


class SkOptOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        """
        Set up for SkOptOptimizerTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}),
                                  PipelineElement("SVC")]
        self.optimizer = SkOptOptimizer()
        self.optimizer_name = "sk_opt"

    def test_ask_advanced(self):
        with self.assertRaises(ValueError):
            super(SkOptOptimizerTest, self).test_ask_advanced()
