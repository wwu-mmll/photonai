import numpy as np
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import ShuffleSplit

from photonai.base import PipelineElement, Hyperpipe
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.optimization import DummyPerformanceConstraint, MinimumPerformanceConstraint, GridSearchOptimizer
from photonai.optimization.optimization_info import Optimization
from photonai.processing.outer_folds import OuterFoldManager
from photonai.processing.photon_folds import FoldInfo
from photonai.processing.results_structure import MDBOuterFold
from photonai.helper.photon_base_test import PhotonBaseTest


class OuterFoldTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(OuterFoldTests, cls).setUpClass()

    def setUp(self):

        super(OuterFoldTests, self).setUp()
        self.fold_nr_inner_cv = 5
        self.inner_cv = ShuffleSplit(n_splits=self.fold_nr_inner_cv, random_state=42)
        self.outer_cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        self.cv_info = Hyperpipe.CrossValidation(inner_cv=self.inner_cv,
                                                 outer_cv=self.outer_cv,
                                                 use_test_set=True,
                                                 test_size=0.2,
                                                 calculate_metrics_per_fold=True,
                                                 calculate_metrics_across_folds=False,
                                                 learning_curves=False,
                                                 learning_curves_cut=None)

        self.X, self.y = load_boston(return_X_y=True)
        self.outer_fold_id = "TestFoldOuter1"
        self.cv_info.outer_folds = {self.outer_fold_id: FoldInfo(0, 1, train, test) for train, test in
                                    self.outer_cv.split(self.X, self.y)}

        self.config_num = 2
        self.optimization_info = Optimization(metrics=['mean_absolute_error', 'mean_squared_error'],
                                              best_config_metric='mean_absolute_error',
                                              optimizer_input='grid_search', optimizer_params={},
                                              performance_constraints=None)
        self.elements = [PipelineElement('StandardScaler'),
                         PipelineElement('PCA', {'n_components': [4, 7]}),
                         PipelineElement('DecisionTreeRegressor', random_state=42)]
        self.pipe = PhotonPipeline([(p.name, p) for p in self.elements])

    def prepare_and_fit(self, outer_fold_man=None):
        if outer_fold_man is None:
            outer_fold = MDBOuterFold(fold_nr=1)
            outer_fold_man = OuterFoldManager(self.pipe, self.optimization_info, self.outer_fold_id, self.cv_info,
                                              result_obj=outer_fold)
        outer_fold_man.fit(self.X, self.y)
        return outer_fold_man

    def test_fit(self):
       self.prepare_and_fit()

    def test_current_best_config(self):

        def check_current_best_config_equality(outer_manager, fold_operation):
            # we know that the first of the two configs is better
            # the value in outer_manager.current_best_config should be the mean value of the best config
            # for train
            self.assertEqual(str(outer_manager.result_object.best_config.get_train_metric(self.optimization_info.best_config_metric,
                                                                                          fold_operation)),
                             str(outer_manager.current_best_config.get_train_metric(self.optimization_info.best_config_metric,
                                                                                    fold_operation)))
            # and for test
            self.assertEqual(str(outer_manager.result_object.best_config.get_test_metric(self.optimization_info.best_config_metric,
                                                                                         fold_operation)),
                             str(outer_manager.current_best_config.get_test_metric(self.optimization_info.best_config_metric,
                                                                                   fold_operation)))

        # if we have calculate_metrics_per_fold = True then
        # we take the mean value for evaluating the current best metric
        self.cv_info.calculate_metrics_across_folds = False
        self.cv_info.calculate_metrics_per_fold = True
        outer_manager_per_fold = self.prepare_and_fit()
        check_current_best_config_equality(outer_manager_per_fold, "mean")

        # if we have calculate_metrics_across_fold = True and cm_per_fold = False, we take the raw value to display
        self.cv_info.calculate_metrics_across_folds = True
        self.cv_info.calculate_metrics_per_fold = False
        outer_manager_across_folds = self.prepare_and_fit()
        check_current_best_config_equality(outer_manager_across_folds, "raw")

    def test_prepare(self):
        self.optimization_info.performance_constraints = [DummyPerformanceConstraint(self.optimization_info.best_config_metric),
                                                          MinimumPerformanceConstraint('mean_squared_error', 75)]
        outer_fold_man = OuterFoldManager(self.pipe, self.optimization_info, self.outer_fold_id, self.cv_info,
                                          result_obj=MDBOuterFold(fold_nr=1))

        outer_fold_man._prepare_optimization()
        outer_fold_man._prepare_data(self.X, self.y)
        # test that performance constraints are copies
        self.assertTrue(outer_fold_man.constraint_objects, list)
        self.assertTrue(len(outer_fold_man.constraint_objects) == 2)
        for ico, copied_object in enumerate(outer_fold_man.constraint_objects):
            self.assertIsNot(self.optimization_info.performance_constraints[ico], copied_object)

        # test that optimizer is prepared and can generated our two configs
        self.assertIsNotNone(outer_fold_man.optimizer)
        self.assertTrue(outer_fold_man.optimizer, GridSearchOptimizer)
        self.assertTrue(len(list(outer_fold_man.optimizer.ask)) == 2)

        # assure that we assured there are no cython leftovers in result tree
        self.assertEqual(len(outer_fold_man.result_object.tested_config_list), 0)

        # test that data is split (we only check y because the split method is already tested, we just make sure it is applied)
        nr_train = len(self.cv_info.outer_folds[self.outer_fold_id].train_indices)
        self.assertTrue(len(outer_fold_man._validation_y) == nr_train)
        nr_test = len(self.cv_info.outer_folds[self.outer_fold_id].test_indices)
        self.assertTrue(len(outer_fold_man._test_y) == nr_test)

        # test that infos are in tree
        self.assertEqual(outer_fold_man.result_object.number_samples_validation, nr_train)
        self.assertEqual(outer_fold_man.result_object.number_samples_test, nr_test)

    def test_eval_final_performance(self):
        # check that best_config_score is copied and not computed
        def case_check(outer_fold_man, operation="mean"):
            # we do so by asserting the values that are in the test set position are EXCACTLY the
            # same as in the values from the validation set in the inner folds (hence, copied, not computed)
            for m_name, m_val in outer_fold_man.result_object.best_config.best_config_score.validation.metrics.items():
                compare_value = outer_fold_man.result_object.best_config.get_test_metric(name=m_name,
                                                                                         operation=operation)
                self.assertAlmostEqual(m_val, compare_value)

            # additionally make sure that there are no predictions and no indices for the test set
            self.assertTrue(len(outer_fold_man.result_object.best_config.best_config_score.validation.indices) == 0)
            self.assertTrue(len(outer_fold_man.result_object.best_config.best_config_score.validation.y_pred) == 0)

        # in case we don't evaluate the test set
        self.cv_info.use_test_set = False
        # todo: metric refactoring update
        # we copy the mean value
        self.cv_info.calculate_metrics_across_folds = False
        self.cv_info.calculate_metrics_per_fold = True
        outer_fold_man = self.prepare_and_fit()
        case_check(outer_fold_man)

        # we still copy the mean value
        self.cv_info.calculate_metrics_across_folds = True
        self.cv_info.calculate_metrics_per_fold = True
        outer_fold_man = self.prepare_and_fit()
        case_check(outer_fold_man)

        # we copy the raw metrics computed across the folds
        self.cv_info.calculate_metrics_across_folds = True
        self.cv_info.calculate_metrics_per_fold = False
        outer_fold_man = self.prepare_and_fit()
        case_check(outer_fold_man, operation="raw")

    def test_save_predictions_and_feature_importances(self):
        # todo: the best config should have feature importances and predictions, all others shouldn't have them
        #  reuse test_save_predictions() and test_feature_importances() from inner_fold_tests.py
        np.random.seed(42)
        outer_fold_man1 = OuterFoldManager(self.pipe, self.optimization_info, self.outer_fold_id, self.cv_info,
                                           result_obj=MDBOuterFold(fold_nr=1))
        self.prepare_and_fit(outer_fold_man1)

        best_config = outer_fold_man1.result_object.best_config
        self.assertTrue(len(best_config.best_config_score.validation.y_true) == len(
            self.cv_info.outer_folds[self.outer_fold_id].test_indices))
        self.assertTrue(len(best_config.best_config_score.validation.y_pred) == len(
            self.cv_info.outer_folds[self.outer_fold_id].test_indices))
        self.assertTrue(len(best_config.best_config_score.feature_importances) == 7)

        for config in outer_fold_man1.result_object.tested_config_list:
            if config.config_nr == best_config.config_nr:
                for fold_i, fold in enumerate(config.inner_folds):
                    self.assertTrue(np.sum(len(fold.validation.y_pred) == 41))
                    self.assertTrue(np.sum(len(fold.feature_importances) == 7))
            else:
                self.assertTrue(np.sum(len(fold.validation.y_pred) for fold in config.inner_folds) == 0)
                self.assertTrue(np.sum(len(fold.validation.probabilities) for fold in config.inner_folds) == 0)
                self.assertTrue(np.sum(len(fold.validation.indices) for fold in config.inner_folds) == 0)
                self.assertTrue(np.sum(len(fold.feature_importances) for fold in config.inner_folds) == 0)

    def test_find_best_config_always_again(self):
        outer_fold_man1 = self.prepare_and_fit()
        outer_fold_man2 = self.prepare_and_fit()

        # we have different entities
        self.assertTrue(outer_fold_man1 is not outer_fold_man2)

        # and they both found the same configuration
        self.assertDictEqual(outer_fold_man1.result_object.best_config.config_dict,
                             outer_fold_man2.result_object.best_config.config_dict)

        # and they both calculated exactly the same values for inner_cv and test set
        self.assertListEqual([str(m) for m in outer_fold_man1.result_object.best_config.metrics_train],
                             [str(m) for m in outer_fold_man2.result_object.best_config.metrics_train])

        self.assertListEqual([str(m) for m in outer_fold_man1.result_object.best_config.metrics_test],
                             [str(m) for m in outer_fold_man2.result_object.best_config.metrics_test])

        self.assertDictEqual(outer_fold_man1.result_object.best_config.best_config_score.validation.metrics,
                             outer_fold_man2.result_object.best_config.best_config_score.validation.metrics,)

        self.assertDictEqual(outer_fold_man1.result_object.best_config.best_config_score.training.metrics,
                             outer_fold_man2.result_object.best_config.best_config_score.training.metrics, )

    def test_fit_dummy(self):
        self.optimization_info.performance_constraints = DummyPerformanceConstraint(self.optimization_info.best_config_metric)
        outer_fold_man = OuterFoldManager(self.pipe, self.optimization_info, self.outer_fold_id, self.cv_info,
                                          result_obj=MDBOuterFold(fold_nr=1))

        outer_fold_man._prepare_optimization()

        # check skipping if no dummy_estimator is given
        outer_fold_man._prepare_data(self.X, self.y)
        outer_fold_man._fit_dummy()
        self.assertIsNone(outer_fold_man.result_object.dummy_results)

        # check for too much dimensions
        outer_fold_man._prepare_data(np.ones((self.X.shape[0], self.X.shape[1], 1)), self.y)
        outer_fold_man._fit_dummy()
        self.assertIsNone(outer_fold_man.result_object.dummy_results)

        # check that dummy result exists with the correct values
        outer_fold_man.dummy_estimator = DummyRegressor()
        outer_fold_man._prepare_data(self.X, self.y)
        outer_fold_man._fit_dummy()

        # for boston housing we expect
        train_values = {'mean_absolute_error': 6.809283403587883, 'mean_squared_error': 86.87340383295755}
        test_values = {'mean_absolute_error': 6.255843525529023, 'mean_squared_error': 75.04543037399255}

        self.assertDictEqual(outer_fold_man.result_object.dummy_results.validation.metrics, test_values)
        self.assertDictEqual(outer_fold_man.result_object.dummy_results.training.metrics, train_values)

        # check that performance constraints are updated
        self.assertTrue(outer_fold_man.constraint_objects[0].threshold ==
                        outer_fold_man.result_object.dummy_results.validation.metrics[self.optimization_info.best_config_metric])
