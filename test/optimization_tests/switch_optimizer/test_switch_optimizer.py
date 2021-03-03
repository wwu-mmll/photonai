from photonai.base import PipelineElement, Switch
from photonai.optimization import FloatRange, IntegerRange
from photonai.optimization.switch_optimizer.meta_optimizer import MetaHPOptimizer
from sklearn.datasets import load_breast_cancer
from ..grid_search_tests.test_grid_search import GridSearchOptimizerTest


class SwitchOptimizerTest(GridSearchOptimizerTest):

    def setUp(self):
        self.pipeline_elements = [PipelineElement("StandardScaler"),
                                  PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)}),
                                  Switch('estimators', [PipelineElement("SVC",
                                                                        {'C': FloatRange(0.1, 1, step=0.2)}),
                                                        PipelineElement('RandomForestClassifier',
                                                                        {'min_samples_split': IntegerRange(2, 5)})])
                                  ]
        self.optimizer = MetaHPOptimizer(name='random_grid_search')
        self.optimizer_name = "switch"
        self.num_configs = 3
        self.optimizer_params = {'name': 'sk_opt', 'n_configurations': self.num_configs}

    def test_wrong_setup(self):
        with self.assertRaises(ValueError):
            _ = MetaHPOptimizer(any_param_but_no_name=1)

        opt = MetaHPOptimizer(name='grid_search')
        with self.assertRaises(ValueError):
            opt.prepare([object()], True)

        opt = MetaHPOptimizer(name='random_name')
        with self.assertRaises(ValueError):
            opt.prepare([Switch('the_switch')], True)

    def test_one_opt_per_estimator(self):
        self.create_hyperpipe()
        for p in self.pipeline_elements:
            self.hyperpipe += p
        X, y = load_breast_cancer(True)
        self.hyperpipe.fit(X, y)

        # check there are three tested configs for each estimator
        self.assertEqual(len(self.hyperpipe.results.outer_folds[0].tested_config_list), 2*self.num_configs)

    def test_ask_advanced(self):
        pass
        #
        # with self.assertRaises(ValueError):
        #     super(SwitchOptimizerTest, self).test_ask_advanced()
