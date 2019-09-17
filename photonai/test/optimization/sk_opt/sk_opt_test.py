import unittest
import types

from photonai.base import PipelineElement
from photonai.optimization import SkOptOptimizer, IntegerRange


class SkOptOptimizerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for SkOptOptimizerTest.
        """
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                  test_disabled=True),
                                  PipelineElement("SVC")]
        self.scikit_optimizer = SkOptOptimizer()

    def test_all_attributes_available(self):
        """
        Test for .ask attribute. .ask is important for next configuration that should be tested.
        """
        self.scikit_optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        self.assertIsInstance(self.scikit_optimizer.ask, types.GeneratorType)
