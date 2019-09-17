import unittest
import types
import numpy as np

from photonai.base import Stack, Switch, PipelineElement
from photonai.optimization import GridSearchOptimizer, RandomGridSearchOptimizer, TimeBoxedRandomGridSearchOptimizer, \
    IntegerRange


class GridSearchOptimizerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for GridSearchTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                  test_disabled=True),
                                  PipelineElement("SVC")]
        self.grid_search_optimizer = GridSearchOptimizer()

    def test_all_attributes_available(self):
        """
        Test for .ask and .param_grid attribute. .ask is important for next configuration that should be tested.
        """
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertIsInstance(self.grid_search_optimizer.ask, types.GeneratorType)
        self.assertIsInstance(self.grid_search_optimizer.param_grid, list)


class RandomGridSearchOptimizerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for GridSearchTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                  test_disabled=True),
                                  PipelineElement("SVC")]
        self.grid_search_optimizer = RandomGridSearchOptimizer()

    def test_all_attributes_available(self):
        """
        Test for .ask and .param_grid attribute. .ask is important for next configuration that should be tested.
        """
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertIsInstance(self.grid_search_optimizer.ask, types.GeneratorType)
        self.assertIsInstance(self.grid_search_optimizer.param_grid, list)

    def test_parameter_k(self):
        """

        """
        self.grid_search_optimizer = RandomGridSearchOptimizer(k=3)
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertEqual(len(self.grid_search_optimizer.param_grid), 3)
        self.grid_search_optimizer = RandomGridSearchOptimizer(k=500)
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertEqual(len(self.grid_search_optimizer.param_grid), 16)


class TimeBoxedRandomGridSearchOptimizerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for GridSearchTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                  test_disabled=True),
                                  PipelineElement("SVC")]
        self.grid_search_optimizer = TimeBoxedRandomGridSearchOptimizer()

    def test_all_attributes_available(self):
        """
        Test for .ask and .param_grid attribute. .ask is important for next configuration that should be tested.
        """
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertIsInstance(self.grid_search_optimizer.ask, types.GeneratorType)
        self.assertIsInstance(self.grid_search_optimizer.param_grid, list)

    def test_parameter_k(self):
        """

        :return:
        """
        self.grid_search_optimizer = RandomGridSearchOptimizer(k=3)
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertEqual(len(self.grid_search_optimizer.param_grid), 3)
        self.grid_search_optimizer = RandomGridSearchOptimizer(k=500)
        self.grid_search_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertEqual(len(self.grid_search_optimizer.param_grid), 16)
