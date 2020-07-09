from photonai.base import PipelineElement
from photonai.optimization import SkOptOptimizer, IntegerRange
from ..grid_search.test_grid_search import GridSearchOptimizerTest


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
