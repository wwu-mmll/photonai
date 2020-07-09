from photonai.base import PipelineElement
from photonai.optimization import RandomSearchOptimizer, IntegerRange
from ..grid_search.test_grid_search import GridSearchOptimizerTest


class RandomSearchOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        """
        Set up for RandomGridSearchOptimizer.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}),
                                  PipelineElement("SVC")]
        self.optimizer = RandomSearchOptimizer(n_configurations=5)
        self.optimizer_name = 'random_search'

    def test_parameter_k(self):
        """
        Test for parameter k.
        """
        self.optimizer = RandomSearchOptimizer(n_configurations=3)
        self.optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        configs = []
        for config in self.optimizer.ask:
            configs.append(config)
        self.assertEqual(len(configs), 3)
        self.optimizer = RandomSearchOptimizer(n_configurations=500)
        self.optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        configs = []
        for config in self.optimizer.ask:
            configs.append(config)
        self.assertEqual(len(configs), 500)

    def test_run(self):
        pass
