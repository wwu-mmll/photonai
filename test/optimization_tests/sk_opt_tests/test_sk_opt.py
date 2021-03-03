from photonai.base import PipelineElement
from photonai.optimization import SkOptOptimizer, IntegerRange, Categorical, FloatRange, BooleanSwitch
from photonai.optimization.hyperparameters import NumberRange
from ..grid_search_tests.test_grid_search import GridSearchOptimizerTest

import warnings
import time


class SkOptOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}),
                                  PipelineElement("SVC", hyperparameters={'C': FloatRange(1, 100)})]
        self.optimizer = SkOptOptimizer()
        self.optimizer_name = "sk_opt"
        self.optimizer_params = None

    def test_ask_advanced(self):
        with self.assertRaises(ValueError):
            super(SkOptOptimizerTest, self).test_ask_advanced()

    def test_empty_hspace(self):
        with warnings.catch_warnings(record=True) as w:
            self.optimizer.prepare([], True)
            self.assertIsNone(self.optimizer.optimizer)
            assert any("Did not find any" in s for s in [e.message.args[0] for e in w])

    def test_eliminate_one_value_hyperparams(self):
        pipeline_elements = [PipelineElement('PCA', hyperparameters={'n_components': Categorical([5])}),
                             PipelineElement("SVC", hyperparameters={'kernel': ['rbf'],
                                                                     'shrinking': BooleanSwitch(),
                                                                     'C':FloatRange(0.5, 2, range_type='linspace'),
                                                                     'tol': FloatRange(0.1, 1, range_type='logspace')})]
        with warnings.catch_warnings(record=True) as w:
            self.optimizer.prepare(pipeline_elements, True)
            assert any("PHOTONAI has detected some" in s for s in [e.message.args[0] for e in w])
        self.assertIn('SVC__C', self.optimizer.hyperparameter_list)
        self.assertIn('SVC__shrinking', self.optimizer.hyperparameter_list)
        self.assertNotIn('PCA__n_components', self.optimizer.hyperparameter_list)
        self.assertNotIn('SVC__kernel', self.optimizer.hyperparameter_list)

        self.optimizer.prepare([pipeline_elements[0]], True)
        self.assertIsNone(self.optimizer.optimizer)

        i = 0
        for val in self.optimizer.ask:
            self.assertFalse(val)
            i += 1
        self.assertEqual(i, 1)

    def test_geomspace(self):
        pipeline_elements = [PipelineElement("SVC", hyperparameters={'kernel': ['rbf', 'poly'],
                                                                     'C': FloatRange(0.1, 0.5, range_type='geomspace'),
                                                                     'tol': FloatRange(0.001, 0.01)})]
        with self.assertRaises(ValueError):
            self.optimizer.prepare(pipeline_elements, True)

    def test_unsported_param(self):
        pipeline_elements = [PipelineElement("SVC", hyperparameters={'kernel': Categorical(['rbf', 'poly', 'sigmoid']),
                                                                     'C': NumberRange(1, 3, range_type='range')})]
        with self.assertRaises(ValueError):
            self.optimizer.prepare(pipeline_elements, True)

    def test_time_limit(self):
        self.optimizer = SkOptOptimizer(limit_in_minutes=0.05, n_configurations=1000)  # 3 seconds
        self.optimizer.prepare(pipeline_elements=self.pipeline_elements, maximize_metric=True)
        configs = []
        start = time.time()
        for config in self.optimizer.ask:
            configs.append(config)
            time.sleep(0.005)
        stop = time.time()
        self.assertAlmostEqual(stop-start, 3, 0)
