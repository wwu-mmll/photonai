
import os
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import PipelineElement, Hyperpipe, OutputSettings, Preprocessing, CallbackElement
from photonai.test.base.photon_pipeline_tests import DummyYAndCovariatesTransformer
from photonai.test.PhotonBaseTest import PhotonBaseTest
from photonai.test.base.photon_elements_tests import elements_to_dict
from photonai.processing.results_structure import MDBConfig, MDBFoldMetric, FoldOperations, MDBInnerFold, MDBOuterFold, MDBScoreInformation


class HyperpipeTests(PhotonBaseTest):

    def setUp(self):

        super(HyperpipeTests, self).setUp()
        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})

        self.inner_cv_object = KFold(n_splits=3)
        self.metrics = ["accuracy", 'recall', 'precision']
        self.best_config_metric = "accuracy"
        self.hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
                                   metrics=self.metrics,
                                   best_config_metric=self.best_config_metric,
                                   output_settings=OutputSettings(project_folder=self.tmp_folder_path))
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_init(self):
        # test that all initi paramters can be retrieved via the cleaned up subclasses
        self.assertEqual(self.hyperpipe.name, 'god')

        # in case don't give infomartion, check for the default parameters, otherwise for the infos given in setUp
        # Cross Validation
        self.assertIsNotNone(self.hyperpipe.cross_validation)
        self.assertEqual(self.hyperpipe.cross_validation.inner_cv, self.inner_cv_object)
        self.assertIsNone(self.hyperpipe.cross_validation.outer_cv, None)
        self.assertTrue(self.hyperpipe.cross_validation.eval_final_performance)
        self.assertTrue(self.hyperpipe.cross_validation.calculate_metrics_per_fold)
        self.assertFalse(self.hyperpipe.cross_validatioin.calculate_metrics_across_folds)
        self.assertIsNone(self.hyperpipe.cross_validation.outer_folds)
        self.assertDictEqual(self.hyperpipe.cross_validation.inner_folds, {})

        # Optimization
        self.assertIsNotNone(self.hyperpipe.optimization)
        self.assertListEqual(self.hyperipe.optimization.metrics, self.metrics)
        self.assertEqual(self.hyperpipe.optimization.best_config_metric, self.best_config_metric)
        self.assertEqual(self.hyperpipe.optimization.optimizer_input_str, "grid_search")
        self.assertTrue(self.hyperpipe.optimization.maximize_metric)
        self.assertIsNone(self.hyperpipe.optimization.performance_constraints)
        self.assertDictEqual(self.hyperpipe.optimization.optimizer_params, {})

    def test_add(self):
        # assure pipeline has two elements, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe._pipe.elements), 3)
        self.assertIs(self.hyperpipe._pipe.elements[0][1], self.ss_pipe_element)
        self.assertIs(self.hyperpipe._pipe.elements[1][1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe._pipe.elements[2][1], self.svc_pipe_element)
        # todo : assure that no two elements can be added with the same name

        # test add method special cases
        with self.assertRaises(TypeError):
            self.hyperpipe.add(object())

        # assure that preprocessing is identified and set to the extra variable, there is only one preprocessing item
        my_preproc = Preprocessing()
        self.assertEqual(my_preproc, self.hyperpipe.preprocessing)
        # make sure the element does not end up in the main pipeline
        self.assertTrue([item[1] is not my_preproc for item in self.hyperpipe.elements])

        def my_func(X, y, **kwargs):
            return True
        # test adding callback item
        my_call_back_item = CallbackElement('test_element', my_func, 'predict')
        self.hyperpipe.add(my_call_back_item)
        self.assertIs(self.hyperpipe.elements[-1][1], my_call_back_item)

    def test_no_metrics(self):
        # make sure that no metrics means raising an error
        with self.assertRaises(ValueError):
            hyperpipe = Hyperpipe("hp_name", inner_cv=self.inner_cv_object)

        # make sure that if no best config metric is given, PHOTON raises a warning
        with self.assertRaises(Warning):
            hyperpipe = Hyperpipe("hp_name", inner_cv=self.inner_cv_object, metrics=["accuracy", "f1_score"])

    def test_preprocessing(self):

        prepro_pipe = Preprocessing()
        prepro_pipe += PipelineElement.create("dummy", DummyYAndCovariatesTransformer(), {})

        self.hyperpipe += prepro_pipe
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(np.array_equal(self.__y + 1, self.hyperpipe.data.y))

    def test_estimation_type(self):
        def callback(X, y=None, **kwargs):
            pass

        pipe = Hyperpipe('name', inner_cv=KFold(n_splits=2), best_config_metric='mean_squared_error')

        with self.assertRaises(NotImplementedError):
            pipe += PipelineElement('PCA')
            est_type = pipe.estimation_type

        pipe += PipelineElement('SVC')
        self.assertEqual(pipe.estimation_type, 'classifier')

        pipe.elements[-1] = PipelineElement('SVR')
        self.assertEqual(pipe.estimation_type, 'regressor')

        with self.assertRaises(NotImplementedError):
            pipe.elements[-1] = CallbackElement('MyCallback', callback)
            est_type = pipe.estimation_type

    def test_copy_me(self):
        self.maxDiff = None
        copy = self.hyperpipe.copy_me()
        copy2 = self.hyperpipe.copy_me()
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(self.hyperpipe))

        copy_after_fit = self.hyperpipe.fit(self.__X, self.__y).copy_me()

        copy_after_fit = elements_to_dict(copy_after_fit)
        # the current_configs of the elements are not None after calling fit() on a hyperpipe
        # when copying the respective PipelineElement, these current_configs are copied, too
        # this is why we need to delete _pipe and elements before asserting for equality
        copy_after_fit['_pipe'] = None
        copy_after_fit['elements'] = None
        copy = elements_to_dict(copy)
        copy['_pipe'] = None
        copy['elements'] = None
        self.assertDictEqual(copy, copy_after_fit)

        # check if deepcopy worked
        copy2.cross_validation.inner_cv.n_splits = 10
        self.assertEqual(copy2.cross_validation.inner_cv.n_splits, 10)
        self.assertEqual(self.hyperpipe.cross_validation.inner_cv.n_splits, 3)

    def recursive_assertion(self, element_a, element_b):
        if isinstance(element_a, dict):
            for key in element_a.keys():
                self.recursive_assertion(element_a[key], element_b[key])
        elif isinstance(element_a, np.ndarray):
            np.testing.assert_array_equal(element_a, element_b)
        elif isinstance(element_a, list):
            for i, _ in enumerate(element_a):
                self.recursive_assertion(element_a[i], element_b[i])
        elif isinstance(element_a, tuple):
            for i in range(len(element_a)):
                self.recursive_assertion(element_a[i], element_b[i])
        else:
            self.assertEqual(element_a, element_b)

    def test_save_optimum_pipe(self):
        # todo
        #  fix is_transformer and is_estimator
        #  redo ModelPersistor and accommodate for Branches and so on (especially Custom Elements within Branches and so on)
        preproc = Preprocessing()
        preproc += PipelineElement('LabelEncoder')
        self.hyperpipe += preproc
        self.hyperpipe.name = "hyperpipe"
        self.hyperpipe.output_settings.project_folder = "./tmp/optimum_pipypipe/"
        pipe_copy = self.hyperpipe.copy_me()
        pipe_copy.output_settings.save_output = False
        pipe_copy.fit(self.__X, self.__y)
        self.assertFalse(os.path.exists("./tmp/optimum_pipypipe/hyperpipe_results/"))

        self.hyperpipe.fit(self.__X, self.__y)
        self.assertTrue(os.path.exists("./tmp/optimum_pipypipe/hyperpipe_results/photon_best_model.photon"))

        # check if load_optimum_pipe also works
        optimum_pipe = Hyperpipe.load_optimum_pipe("./tmp/optimum_pipypipe/hyperpipe_results/photon_best_model.photon")
        self.recursive_assertion(elements_to_dict(optimum_pipe), elements_to_dict(self.hyperpipe.optimum_pipe))

class HyperpipeOptimizationClassTests(unittest.TestCase):

    def test_best_config_metric(self):
        my_pipe_optimizer = Hyperpipe.Optimization('grid_search', {}, [], 'balanced_accuracy', None)
        self.assertTrue(my_pipe_optimizer.maximize_metric)
        my_pipe_optimizer = Hyperpipe.Optimization('grid_search', {}, [], 'mean_squared_error', None)
        self.assertFalse(my_pipe_optimizer.maximize_metric)

    def test_optmizer_input_str(self):
        with self.assertRaises(ValueError):
            my_pipe_optimizer = Hyperpipe.Optimization('unknown_optimizer', {}, [], 'accuracy', None)

        for name, opt_class in Hyperpipe.Optimization.OPTIMIZER_DICTIONARY:
            my_pipe_optimizer = Hyperpipe.Optimization(name, {}, [], 'accuracy', None)
            optimizer_obj = my_pipe_optimizer.get_optimizer()
            self.assertIsInstance(optimizer_obj, opt_class)

    def test_get_optimum_config(self):
        my_pipe_optimizer = Hyperpipe.Optimization('grid_search', {}, [], 'balanced_accuracy', None)
        list_of_tested_configs = list()
        metric_default = MDBFoldMetric(metric_name = 'balanced_accuracy',
                                       operation = FoldOperations.MEAN,
                                       value = 0.5)
        metric_best = MDBFoldMetric(metric_name = 'balanced_accuracy',
                                    operation = FoldOperations.MEAN,
                                    value = 0.99)
        # we add looser configs, one good config, and one good config that failed
        # and check if the good non-failing config is chosen
        for i in range(10):
            config = MDBConfig()
            # number 5 is the winner
            if i == 5 or i==8:
                config.metrics_test = [metric_best]
            else:
                config.metrics_tst = [metric_default]
            if i==8:
                config.config_failed = True
            list_of_tested_configs.append(config)

        winner_config = my_pipe_optimizer.get_optimum_config(list_of_tested_configs)
        self.assertIs(winner_config, list_of_tested_configs[5])
        self.assertEqual(winner_config.metrics_test[0].value, 0.99)

    def test_get_optimum_config_outer_folds(self):
        my_pipe_optimizer = Hyperpipe.Optimization('grid_search', {}, [], 'balanced_accuracy', None)

        outer_fold_list = list()
        for i in range(10):
            outer_fold = MDBOuterFold()
            outer_fold.best_config = MDBConfig()
            outer_fold.best_config.best_config_score = MDBInnerFold()
            outer_fold.best_config.best_config_score.validation = MDBScoreInformation
            # again fold 5 wins
            if i==5:
                outer_fold.best_config.best_config_score.metrics = {'balanced_accuracy' : 0.99}
            else:
                outer_fold.best_config.best_config_score.metrics = {'balanced_accuracy': 0.5}
            outer_fold_list.append(outer_fold)

        best_config_outer_folds = my_pipe_optimizer.get_optimum_config_outer_folds(outer_fold_list)
        self.assertEqual(best_config_outer_folds.best_config_score.metrics['balanced_accuracy'], 0.99)
        self.assertIs(best_config_outer_folds, outer_fold_list[5].best_config)
