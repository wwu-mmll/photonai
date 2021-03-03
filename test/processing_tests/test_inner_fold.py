import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
# stuff for simultaneously training and testing with sklearn
from sklearn.preprocessing import StandardScaler

from photonai.base import PipelineElement, Hyperpipe, OutputSettings
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.optimization import MinimumPerformanceConstraint, FloatRange
from photonai.optimization.optimization_info import Optimization
from photonai.processing.inner_folds import InnerFoldManager
from photonai.processing.photon_folds import FoldInfo
from photonai.helper.photon_base_test import PhotonBaseTest


# ------------------------------------------------------------

class InnerFoldTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(InnerFoldTests, cls).setUpClass()

    def setUp(self):
        super(InnerFoldTests, self).setUp()
        self.pipe = PhotonPipeline([('StandardScaler', PipelineElement('StandardScaler')),
                                    ('PCA', PipelineElement('PCA')),
                                    ('RidgeClassifier', PipelineElement('RidgeClassifier'))])
        self.config = {'PCA__n_components': 5, 'RidgeClassifier__solver': 'svd', 'RidgeClassifier__random_state': 42}
        self.outer_fold_id = 'TestID'
        self.inner_cv = KFold(n_splits=4)
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.cross_validation = Hyperpipe.CrossValidation(self.inner_cv, None, True, 0.2, True, False, False, None)
        self.cross_validation.inner_folds = {self. outer_fold_id: {i: FoldInfo(i, i+1, train, test) for i, (train, test) in
                                                                   enumerate(self.inner_cv.split(self.X, self.y))}}
        self.optimization = Optimization('grid_search', {}, ['accuracy', 'recall', 'specificity'], 'accuracy', None)

    def test_fit_against_sklearn(self):
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id)

        photon_results_config_item = test_pipe.fit(self.X, self.y)
        self.assertIsNotNone(photon_results_config_item.computation_start_time)
        self.assertIsNotNone(photon_results_config_item.computation_end_time)

        # now sklearn.
        sklearn_pipe = Pipeline([('StandardScaler', StandardScaler()),
                                 ('PCA', PCA()),
                                 ('RidgeClassifier', RidgeClassifier())])
        sklearn_pipe.set_params(**self.config)
        for fold_obj in self.cross_validation.inner_folds[self.outer_fold_id].values():
            train_X, test_X = self.X[fold_obj.train_indices], self.X[fold_obj.test_indices]
            train_y, test_y = self.y[fold_obj.train_indices], self.y[fold_obj.test_indices]

            sklearn_pipe.fit(train_X, train_y)
            sklearn_predictions = sklearn_pipe.predict(test_X)
            sklearn_feature_importances = sklearn_pipe.named_steps['RidgeClassifier'].coef_

            photon_test_results = photon_results_config_item.inner_folds[fold_obj.fold_nr - 1].validation

            self.assertTrue(np.array_equal(sklearn_predictions, photon_test_results.y_pred))

            for fi, sklearn_feature_importance_score in enumerate(sklearn_feature_importances[0]):
                self.assertAlmostEqual(sklearn_feature_importance_score,
                                       photon_results_config_item.inner_folds[fold_obj.fold_nr - 1].feature_importances[
                                           0][fi])

            accuracy = accuracy_score(test_y, sklearn_predictions)
            self.assertEqual(photon_test_results.metrics['accuracy'], accuracy)

            recall = recall_score(test_y, sklearn_predictions)
            self.assertEqual(photon_test_results.metrics['recall'], recall)

    def test_performance_constraints(self):
        # test if the constraints are considered
        # A: for a single constraint
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id,
                                     optimization_constraints=MinimumPerformanceConstraint('accuracy', 0.95, 'first'))

        photon_results_config_item = test_pipe.fit(self.X, self.y)
        # the first fold has an accuracy of 0.874 so we expect the test_pipe to stop calculating after the first fold
        # which means it has only one outer fold and
        self.assertTrue(len(photon_results_config_item.inner_folds) == 1)

        # B: for a list of constraints, accuracy should pass (0.874 in first fold > accuracy threshold)
        # but specificity should stop the computation (0.78 in first fold < specificity threshold)
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id,
                                     optimization_constraints=[MinimumPerformanceConstraint('accuracy', 0.85, 'first'),
                                                               MinimumPerformanceConstraint('specificity', 0.8, 'first')])

        photon_results_config_item = test_pipe.fit(self.X, self.y)
        self.assertTrue(len(photon_results_config_item.inner_folds) == 1)

        # C: for a list of constraints, all should pass
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id,
                                     optimization_constraints=[MinimumPerformanceConstraint('accuracy', 0.75, 'any'),
                                                               MinimumPerformanceConstraint('specificity', 0.75, 'any')])

        photon_results_config_item = test_pipe.fit(self.X, self.y)
        self.assertTrue(len(photon_results_config_item.inner_folds) == 4)

    def test_raise_error(self):

        # case A: raise_error = False -> we expect continuation of the computation
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id,
                                     raise_error=False)

        # computing with inequal number of features and targets should result in an error
        test_pipe.fit(self.X, self.y[:10])

        # case B:
        test_pipe.raise_error = True
        with self.assertRaises(IndexError):
            test_pipe.fit(self.X, self.y[:10])

    def test_save_predictions(self):

        # assert that we have the predictions stored
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id)

        # in case we want to have metrics calculated across false, we need to temporarily store the predictions
        test_pipe.optimization_infos.calculate_metrics_across_folds = True
        config_item = test_pipe.fit(self.X, self.y)

        for inner_fold in config_item.inner_folds:
            self.assertEqual(len(inner_fold.training.y_pred), inner_fold.number_samples_training)
            self.assertEqual(len(inner_fold.validation.y_pred), inner_fold.number_samples_validation)

    def test_save_feature_importances(self):
        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id)

        # we expect the feature importances to be of length 5 because the input is through the PCA reduced to 5 dimensions
        output_config = test_pipe.fit(self.X, self.y)
        for inner_fold in output_config.inner_folds:
            self.assertEqual(len(inner_fold.feature_importances[0]), 5)

    def test_process_fit_results(self):

        test_pipe = InnerFoldManager(self.pipe.copy_me, self.config, self.optimization,
                                     self.cross_validation, self.outer_fold_id)
        test_pipe.cross_validation_infos.calculate_metrics_across_folds = True
        test_pipe.cross_validation_infos.calculate_metrics_per_fold = False
        across_folds_config_item = test_pipe.fit(self.X, self.y)

        test_pipe.cross_validation_infos.calculate_metrics_across_folds = False
        test_pipe.cross_validation_infos.calculate_metrics_per_fold = True
        per_fold_config_item = test_pipe.fit(self.X, self.y)

        test_pipe.cross_validation_infos.calculate_metrics_across_folds = True
        test_pipe.cross_validation_infos.calculate_metrics_per_fold = True
        across_and_per_folds_config_item = test_pipe.fit(self.X, self.y)

        def assert_fold_operations(expected_operations, returned_metric_list):
            # assert that we have raw and std and mean
            expected_returns = list()
            for metric in self.optimization.metrics:
                for operation in expected_operations:
                    expected_returns.append(metric + "__" + str(operation))

            returned_formatted_metric_list = [m.metric_name + "__" + str(m.operation) for m in returned_metric_list]
            self.assertTrue(set(expected_returns) == set(returned_formatted_metric_list))

        # if we have both, then we have mean and std over the folds + three raw across folds
        num_of_metrics = len(test_pipe.optimization_infos.metrics)
        self.assertTrue(len(across_and_per_folds_config_item.metrics_train) == 2 * num_of_metrics + num_of_metrics)
        self.assertTrue(len(across_and_per_folds_config_item.metrics_test) == 2 * num_of_metrics + num_of_metrics)

        assert_fold_operations(["raw", "mean", "std"],
                               across_and_per_folds_config_item.metrics_train)
        assert_fold_operations(["raw", "mean", "std"],
                               across_and_per_folds_config_item.metrics_test)

        # if we have across folds only, then it should be 3, one for each metrics
        self.assertTrue(len(across_folds_config_item.metrics_train) == num_of_metrics)
        self.assertTrue(len(across_folds_config_item.metrics_test) == num_of_metrics)

        assert_fold_operations(["raw"], across_folds_config_item.metrics_train)
        assert_fold_operations(["raw"], across_folds_config_item.metrics_test)

        # if we have per fold only, then it should be 6, one for mean and std for each of the three metrics
        self.assertTrue(len(per_fold_config_item.metrics_train) == 2 * num_of_metrics)
        self.assertTrue(len(per_fold_config_item.metrics_test) == 2 * num_of_metrics)
        assert_fold_operations(["mean", "std"], per_fold_config_item.metrics_train)
        assert_fold_operations(["mean", "std"], per_fold_config_item.metrics_test)

    def test_extract_feature_importances(self):
        # one machine with coef_
        self.pipe.fit(self.X, self.y)
        f_importances_coef = self.pipe.feature_importances_
        self.assertTrue(f_importances_coef is not None)
        self.assertTrue(isinstance(f_importances_coef, list))

        # one machine with feature_importances_
        f_imp_pipe = PhotonPipeline([('StandardScaler', PipelineElement('StandardScaler')),
                                    ('PCA', PipelineElement('PCA')),
                                    ('DecisionTreeClassifier', PipelineElement('DecisionTreeClassifier'))])
        f_imp_pipe.fit(self.X, self.y)
        f_importances = f_imp_pipe.feature_importances_
        self.assertTrue(f_importances is not None)
        self.assertTrue(isinstance(f_importances, list))

        # one machine that has no feature importances
        no_f_imp_pipe = PhotonPipeline([('StandardScaler', PipelineElement('StandardScaler')),
                                        ('PCA', PipelineElement('PCA')),
                                        ('SVC', PipelineElement('SVC', kernel='rbf'))])
        no_f_imp_pipe.fit(self.X, self.y)
        no_f_imps = no_f_imp_pipe.feature_importances_
        self.assertTrue(no_f_imps is None )

    def test_learning_curves(self):
        def test_one_hyperpipe(learning_curves, learning_curves_cut):
            if learning_curves and learning_curves_cut is None:
                learning_curves_cut = FloatRange(0, 1, 'range', 0.2)
            output_settings = OutputSettings(save_output=False)
            test_hyperpipe = Hyperpipe('test_pipe',
                                       learning_curves=learning_curves,
                                       learning_curves_cut=learning_curves_cut,
                                       metrics=['accuracy', 'recall', 'specificity'],
                                       best_config_metric='accuracy',
                                       inner_cv=self.inner_cv,
                                       project_folder=self.tmp_folder_path,
                                       output_settings=output_settings)

            self.assertEqual(test_hyperpipe.cross_validation.learning_curves, learning_curves)
            if learning_curves:
                self.assertEqual(test_hyperpipe.cross_validation.learning_curves_cut, learning_curves_cut)
            else:
                self.assertIsNone(test_hyperpipe.cross_validation.learning_curves_cut)

            test_hyperpipe += PipelineElement('StandardScaler')
            test_hyperpipe += PipelineElement('PCA', {'n_components': [1, 2]}, random_state=42)
            test_hyperpipe += PipelineElement('SVC', {'C': [0.1], 'kernel': ['linear']}, random_state=42)
            test_hyperpipe.fit(self.X, self.y)
            config_results = test_hyperpipe.results_handler.results.outer_folds[0].tested_config_list
            config_num = len(config_results)
            for config_nr in range(config_num):
                for inner_fold_nr in range(self.inner_cv.n_splits):
                    curves = config_results[config_nr].inner_folds[inner_fold_nr].learning_curves
                    if learning_curves:
                        self.assertEqual(len(curves), len(learning_curves_cut.values))
                        for learning_point_nr in range(len(learning_curves_cut.values)):
                            test_metrics = list(curves[learning_point_nr][1].keys())
                            train_metrics = list(curves[learning_point_nr][2].keys())
                            self.assertEqual(test_hyperpipe.optimization.metrics, test_metrics)
                            self.assertEqual(test_hyperpipe.optimization.metrics, train_metrics)
                    else:
                        self.assertEqual(curves, [])
        # hyperpipe with properly set learning curves
        test_one_hyperpipe(learning_curves=True, learning_curves_cut=FloatRange(0, 1, 'range', 0.5))
        # hyperpipe without cut (default cut should be used here)
        test_one_hyperpipe(learning_curves=True, learning_curves_cut=None)
        # hyperpipe with cut despite learning_curves being False
        test_one_hyperpipe(learning_curves=False, learning_curves_cut=FloatRange(0, 1, 'range', 0.5))
