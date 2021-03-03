import time

from photonai.base import PipelineElement
from photonai.optimization import RandomSearchOptimizer, IntegerRange
from ..grid_search_tests.test_grid_search import GridSearchOptimizerTest


class RandomSearchOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}),
                                  PipelineElement("SVC")]
        self.optimizer = RandomSearchOptimizer(n_configurations=5)
        self.optimizer_name = 'random_search'

    def test_parameter_k(self):
        """Test for parameter n_configuration and k."""
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

    def test_constraint_obligation(self):
        with self.assertRaises(ValueError):
            RandomSearchOptimizer(n_configurations=-1, limit_in_minutes=-1)

    def test_time_limit(self):
        self.optimizer = RandomSearchOptimizer(limit_in_minutes=0.05)  # 3 seconds
        self.optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        configs = []
        start = time.time()
        for config in self.optimizer.ask:
            configs.append(config)
        stop = time.time()
        self.assertAlmostEqual(stop-start, 3, 0)

    def test_run(self):
        pass
