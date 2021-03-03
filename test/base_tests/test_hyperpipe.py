import datetime
import os
import shutil
import time
import unittest
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from keras.metrics import Accuracy

from photonai.base import PipelineElement, Hyperpipe, OutputSettings, Preprocessing, CallbackElement, Branch, Stack, \
    Switch, ParallelBranch
from photonai.optimization import IntegerRange, Categorical
from photonai.optimization.optimization_info import Optimization
from photonai.processing.results_handler import ResultsHandler
from photonai.processing.results_structure import MDBConfig, MDBFoldMetric, \
    MDBInnerFold, MDBOuterFold, MDBScoreInformation, MDBDummyResults, MDBHyperpipe
from photonai.helper.dummy_elements import DummyTransformer, DummyYAndCovariatesTransformer
from photonai.helper.photon_base_test import elements_to_dict, PhotonBaseTest


class HyperpipeTests(PhotonBaseTest):

    def setup_hyperpipe(self, output_settings=None):
        self.hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
                                   metrics=self.metrics,
                                   best_config_metric=self.best_config_metric,
                                   project_folder=self.tmp_folder_path,
                                   output_settings=output_settings,
                                   verbosity=0)
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(HyperpipeTests, cls).setUpClass()

    def setUp(self):

        super(HyperpipeTests, self).setUp()
        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, random_state=42, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['linear']}, # 'rbf', 'sigmoid']
                                                random_state=42)

        self.inner_cv_object = KFold(n_splits=3)
        self.metrics = ["accuracy", 'recall', 'precision']
        self.best_config_metric = "accuracy"
        self.setup_hyperpipe()

        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_init(self):
        # test that all init parameters can be retrieved via the cleaned up subclasses
        self.assertEqual(self.hyperpipe.name, 'god')

        # in case don't give information, check for the default parameters, otherwise for the infos given in setUp
        # Cross Validation
        self.assertIsNotNone(self.hyperpipe.cross_validation)
        self.assertEqual(self.hyperpipe.cross_validation.inner_cv, self.inner_cv_object)
        self.assertIsNone(self.hyperpipe.cross_validation.outer_cv, None)
        self.assertTrue(self.hyperpipe.cross_validation.use_test_set)
        self.assertTrue(self.hyperpipe.cross_validation.calculate_metrics_per_fold)
        self.assertFalse(self.hyperpipe.cross_validation.calculate_metrics_across_folds)
        self.assertIsNone(self.hyperpipe.cross_validation.outer_folds)
        self.assertDictEqual(self.hyperpipe.cross_validation.inner_folds, {})

        # Optimization
        self.assertIsNotNone(self.hyperpipe.optimization)
        self.assertListEqual(self.hyperpipe.optimization.metrics, self.metrics)
        self.assertEqual(self.hyperpipe.optimization.best_config_metric, self.best_config_metric)
        self.assertEqual(self.hyperpipe.optimization.optimizer_input_str, "grid_search")
        self.assertTrue(self.hyperpipe.optimization.maximize_metric)
        self.assertIsNone(self.hyperpipe.optimization.performance_constraints)
        self.assertDictEqual(self.hyperpipe.optimization.optimizer_params, {})

    def test_add(self):
        # assure pipeline has two elements, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe.elements), 3)
        self.assertIs(self.hyperpipe.elements[0], self.ss_pipe_element)
        self.assertIs(self.hyperpipe.elements[1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe.elements[2], self.svc_pipe_element)
        # todo : assure that no two elements can be added with the same name

        # test add method special cases
        with self.assertRaises(TypeError):
            self.hyperpipe.add(object())

        # assure that preprocessing is identified and set to the extra variable, there is only one preprocessing item
        my_preproc = Preprocessing()
        self.hyperpipe.add(my_preproc)
        self.assertEqual(my_preproc, self.hyperpipe.preprocessing)
        # make sure the element does not end up in the main pipeline
        self.assertTrue([item is not my_preproc for item in self.hyperpipe.elements])

        def my_func(X, y, **kwargs):
            return True
        # test adding callback item
        my_call_back_item = CallbackElement('test_element', my_func, 'predict')
        self.hyperpipe.add(my_call_back_item)
        self.assertIs(self.hyperpipe.elements[-1], my_call_back_item)

    def test_sanity(self):
        # make sure that no metrics means raising an error
        with self.assertRaises(ValueError):
            Hyperpipe("hp_name", inner_cv=self.inner_cv_object)

        # make sure that if no best config metric is given, PHOTON raises a warning
        # with self.assertWarns(Warning) as w:
        # with warnings.catch_warnings(record=True) as w:
        with self.assertRaises(Warning) as w:
            Hyperpipe("hp_name", inner_cv=self.inner_cv_object, metrics=["accuracy", "f1_score"])
            assert any("No best config metric was given" in s for s in [e.message.args[0] for e in w])

        # with warnings.catch_warnings(record=True) as w:
        with self.assertRaises(Warning) as w:
            Hyperpipe("hp_name", inner_cv=self.inner_cv_object, best_config_metric=["accuracy", "f1_score"])
            assert any("Best Config Metric must be a single" in s for s in [e.message.args[0] for e in w])

        with self.assertRaises(NotImplementedError):
            Hyperpipe("hp_name", inner_cv=self.inner_cv_object,
                                 best_config_metric='accuracy', metrics=["accuracy"],
                                 calculate_metrics_across_folds=False,
                                 calculate_metrics_per_fold=False)

        with self.assertRaises(AttributeError):
            Hyperpipe("hp_name", best_config_metric='accuracy', metrics=["accuracy"])

        data = np.random.random((500, 50))

        with self.assertRaises(ValueError):
            targets = np.random.randint(0, 1, (500, 2))
            self.hyperpipe.fit(data, targets)

    def test_hyperpipe_with_custom_metric(self):

        def custom_metric(y_true, y_pred):
            return 99.9

        self.hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
                                   metrics=[('custom_metric', custom_metric), 'accuracy'],
                                   best_config_metric=Accuracy,
                                   project_folder=self.tmp_folder_path)
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue('custom_metric' in self.hyperpipe.results.best_config.best_config_score.validation.metrics)
        self.assertEqual(self.hyperpipe.results.best_config.best_config_score.validation.metrics['custom_metric'], 99.9)

        expected_num_of_metrics = len(self.hyperpipe.optimization.metrics)
        # one: accuracy, two: custom metric registered as "custom_metric", three: keras Metric registered as function
        self.assertEqual(expected_num_of_metrics, 3)

        # dummy average values
        self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_train), expected_num_of_metrics)
        self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_test), expected_num_of_metrics)

        # overall average values
        self.assertTrue(len(self.hyperpipe.results.metrics_train), 2 * expected_num_of_metrics)
        self.assertTrue(len(self.hyperpipe.results.metrics_test), 2 * expected_num_of_metrics)

    def test_preprocessing(self):

        prepro_pipe = Preprocessing()
        prepro_pipe += PipelineElement.create("dummy", DummyYAndCovariatesTransformer(), {})

        self.hyperpipe += prepro_pipe
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(np.array_equal(self.__y + 1, self.hyperpipe.data.y))

    def test_permutation_feature_importances(self):
        hp = Hyperpipe('god',
                       inner_cv=self.inner_cv_object,
                       metrics=self.metrics,
                       best_config_metric=self.best_config_metric,
                       project_folder=self.tmp_folder_path,
                       verbosity=0)
        svc = PipelineElement('SVC')
        hp += svc
        hp.fit(self.__X, self.__y)

        score_photon = hp.score(self.__X, self.__y)
        score_element = svc.score(self.__X, self.__y)
        self.assertAlmostEqual(score_photon, score_element)

        permutation_score = hp.get_permutation_feature_importances(self.__X, self.__y, n_repeats=50, random_state=0)
        score_2 = permutation_importance(svc, self.__X, self.__y, n_repeats=50, random_state=0)
        np.testing.assert_array_equal(permutation_score["importances"], score_2["importances"])

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

    def test_save_optimum_pipe(self):
        tmp_path = os.path.join(self.tmp_folder_path, 'optimum_pipypipe')
        settings = OutputSettings(overwrite_results=True)

        my_pipe = Hyperpipe('hyperpipe',
                            optimizer='random_grid_search',
                            optimizer_params={'n_configurations': 3},
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='f1_score',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            verbosity=0,
                            project_folder=tmp_path,
                            output_settings=settings)

        preproc = Preprocessing()
        preproc += PipelineElement('StandardScaler')

        # BRANCH WITH QUANTILTRANSFORMER AND DECISIONTREECLASSIFIER
        tree_qua_branch = Branch('tree_branch')
        tree_qua_branch += PipelineElement('QuantileTransformer')
        tree_qua_branch += PipelineElement('DecisionTreeClassifier', {'min_samples_split': IntegerRange(2, 4)},
                                           criterion='gini')

        # BRANCH WITH MinMaxScaler AND DecisionTreeClassifier
        svm_mima_branch = Branch('svm_branch')
        svm_mima_branch += PipelineElement('MinMaxScaler')
        svm_mima_branch += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']), 'C': 2.0}, gamma='auto')

        # BRANCH WITH StandardScaler AND KNeighborsClassifier
        knn_sta_branch = Branch('neighbour_branch')
        knn_sta_branch += PipelineElement.create("dummy", DummyTransformer(), {})
        knn_sta_branch += PipelineElement('KNeighborsClassifier')

        my_pipe += preproc
        # voting = True to mean the result of every branch
        my_pipe += Stack('final_stack', [tree_qua_branch, svm_mima_branch, knn_sta_branch])

        my_pipe += PipelineElement('LogisticRegression', solver='lbfgs')

        my_pipe.fit(self.__X, self.__y)
        model_path = os.path.join(my_pipe.output_settings.results_folder, 'photon_best_model.photon')
        self.assertTrue(os.path.exists(model_path))

        # now move optimum pipe to new folder
        test_folder = os.path.join(my_pipe.output_settings.results_folder, 'new_test_folder')
        new_model_path = os.path.join(test_folder, 'photon_best_model.photon')
        os.makedirs(test_folder)
        shutil.copyfile(model_path, new_model_path)

        # check if load_optimum_pipe also works
        # check if we have the meta information recovered
        loaded_optimum_pipe = Hyperpipe.load_optimum_pipe(new_model_path)
        self.assertIsNotNone(loaded_optimum_pipe._meta_information)
        self.assertIsNotNone(loaded_optimum_pipe._meta_information['photon_version'])

        # check if predictions stay realiably the same
        y_pred_loaded = loaded_optimum_pipe.predict(self.__X)
        y_pred = my_pipe.optimum_pipe.predict(self.__X)
        np.testing.assert_array_equal(y_pred_loaded, y_pred)

    def test_save_optimum_pipe_custom_element(self):
        tmp_path = os.path.join(self.tmp_folder_path, 'optimum_pipypipe')
        settings = OutputSettings(overwrite_results=True)

        my_pipe = Hyperpipe('hyperpipe',
                            optimizer='random_grid_search',
                            optimizer_params={'n_configurations': 1},
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='f1_score',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            verbosity=0,
                            project_folder=tmp_path,
                            output_settings=settings)
        my_pipe += PipelineElement('KerasDnnClassifier', {}, epochs=1, hidden_layer_sizes=[5])
        my_pipe.fit(self.__X, self.__y)
        model_path = os.path.join(my_pipe.output_settings.results_folder, 'photon_best_model.photon')
        self.assertTrue(os.path.exists(model_path))

        # check if load_optimum_pipe also works
        # check if we have the meta information recovered
        loaded_optimum_pipe = Hyperpipe.load_optimum_pipe(model_path)
        self.assertIsNotNone(loaded_optimum_pipe._meta_information)

    def test_failure_to_save_optimum_pipe(self):
        tmp_path = os.path.join(self.tmp_folder_path, 'optimum_pipypipe')
        settings = OutputSettings(overwrite_results=True)

        my_pipe = Hyperpipe('hyperpipe',
                            optimizer='random_grid_search',
                            optimizer_params={'n_configurations': 1},
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='f1_score',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            verbosity=0,
                            project_folder=tmp_path,
                            output_settings=settings)
        my_pipe += PipelineElement('KNeighborsClassifier')
        my_pipe.fit(self.__X, self.__y)
        model_path = os.path.join(my_pipe.output_settings.results_folder, 'photon_best_model_wrong_path.photon')
        with self.assertRaises(FileNotFoundError):
            Hyperpipe.load_optimum_pipe(model_path)

    def test_overwrite_result_folder(self):
        """
        Test for right handling of parameter output_settings.overwrite.
        """
        def get_summary_file():
            return os.path.join(self.hyperpipe.output_settings.results_folder, 'photon_summary.txt')

        # Case 1: default
        output_settings1 = OutputSettings(save_output=True,
                                          overwrite_results=False)
        self.setup_hyperpipe(output_settings1)
        self.hyperpipe.fit(self.__X, self.__y)
        tmp_path = get_summary_file()

        time.sleep(2)

        # again with same settings
        self.setup_hyperpipe(output_settings1)
        self.hyperpipe.fit(self.__X, self.__y)
        tmp_path2 = get_summary_file()

        # we expect a new output folder each time with timestamp
        self.assertNotEqual(tmp_path, tmp_path2)

        # Case 2 overwrite results: all in the same folder
        output_settings2 = OutputSettings(save_output=True,
                                          overwrite_results=True)
        self.setup_hyperpipe(output_settings2)
        self.hyperpipe.fit(self.__X, self.__y)
        tmp_path = get_summary_file()
        tmp_date = os.path.getmtime(tmp_path)

        self.setup_hyperpipe(output_settings2)
        self.hyperpipe.fit(self.__X, self.__y)
        tmp_path2 = get_summary_file()
        tmp_date2 = os.path.getmtime(tmp_path2)

        # same folder but summary file is overwritten through the new analysis
        self.assertEqual(tmp_path, tmp_path2)
        self.assertNotEqual(tmp_date, tmp_date2)

        # Case 3: we have a cache folder
        self.hyperpipe.cache_folder = self.cache_folder_path
        shutil.rmtree(self.cache_folder_path, ignore_errors=True)
        self.hyperpipe.fit(self.__X, self.__y)
        self.assertTrue(os.path.exists(self.cache_folder_path))

    def test_random_state(self):
        self.hyperpipe.random_state = 4567
        self.hyperpipe.fit(self.__X, self.__y)
        # assure we spread the word.. !
        self.assertEqual(self.hyperpipe.random_state, 4567)
        self.assertEqual(self.hyperpipe._pipe.random_state, 4567)
        self.assertEqual(self.hyperpipe.optimum_pipe.random_state, 4567)
        self.assertEqual(self.hyperpipe._pipe.elements[-1][-1].random_state, 4567)
        self.assertEqual(self.hyperpipe._pipe.elements[-1][-1].base_element.random_state, 4567)


    def test_dummy_estimator_preparation(self):

        self.hyperpipe.results = MDBHyperpipe()
        self.hyperpipe.results.dummy_estimator = dummy_estimator = MDBDummyResults()

        # one time regressor, one time classifier, one time strange object
        self.hyperpipe.elements = list()
        self.hyperpipe.add(PipelineElement('SVC'))
        dummy_estimator = self.hyperpipe._prepare_dummy_estimator()
        self.assertTrue(isinstance(dummy_estimator, DummyClassifier))

        self.hyperpipe.elements = list()
        self.hyperpipe.add(PipelineElement('SVR'))
        dummy_estimator = self.hyperpipe._prepare_dummy_estimator()
        self.assertTrue(isinstance(dummy_estimator, DummyRegressor))

        with self.assertRaises(NotImplementedError):
            self.hyperpipe.elements = list()
            self.hyperpipe.add(PipelineElement('PCA'))
            dummy_estimator = self.hyperpipe._prepare_dummy_estimator()
            self.assertIsNone(dummy_estimator)

    def setup_crazy_pipe(self):
        # erase all, we need a complex and crazy task
        self.hyperpipe.elements = list()

        nmb_list = list()
        for i in range(5):
            nmb = ParallelBranch(name=str(i), nr_of_processes=i+3)
            sp = PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(1, 50)})
            nmb += sp
            nmb_list.append(nmb)

        my_switch = Switch('disabling_test_switch')
        my_switch += nmb_list[0]
        my_switch += nmb_list[1]

        my_stack = Stack('stack_of_branches')
        for i in range(3):
            my_branch = Branch('branch_' + str(i+2))
            my_branch += PipelineElement('StandardScaler')
            my_branch += nmb_list[i+2]
            my_stack += my_branch

        self.hyperpipe.add(my_stack)
        self.hyperpipe.add(PipelineElement('StandardScaler'))
        self.hyperpipe.add(my_switch)
        self.hyperpipe.add(PipelineElement('SVC'))
        return nmb_list

    def test_recursive_disabling(self):
        list_of_elements_to_detect = self.setup_crazy_pipe()
        self.hyperpipe._pipe = Branch.prepare_photon_pipe(list_of_elements_to_detect)
        Hyperpipe.disable_multiprocessing_recursively(self.hyperpipe._pipe)
        self.assertTrue([i.nr_of_processes == 1 for i in list_of_elements_to_detect])

    def test_recursive_cache_folder_propagation(self):
        list_of_elements = self.setup_crazy_pipe()
        self.hyperpipe._pipe = Branch.prepare_photon_pipe(self.hyperpipe.elements)
        self.hyperpipe.recursive_cache_folder_propagation(self.hyperpipe._pipe, self.cache_folder_path, 'fold_id_123')
        for i, nmbranch in enumerate(list_of_elements):
            if i > 1:
                start_folder = os.path.join(self.cache_folder_path, 'branch_' + nmbranch.name)
            else:
                start_folder = self.cache_folder_path
            expected_folder = os.path.join(start_folder, nmbranch.name)
            self.assertEqual(nmbranch.base_element.cache_folder, expected_folder)

    def test_setup_error_file(self):
        # when we init the hyperpipe the file should exist
        self.assertTrue(os.path.isfile(self.hyperpipe.output_settings.setup_error_file))
        # when we call fit it shall disappear
        # we call it empty so that the computation does not occur
        with self.assertRaises(ValueError):
            self.hyperpipe.fit(None, None)
        # however the file should be gone by now
        self.assertFalse(os.path.isfile(self.hyperpipe.output_settings.setup_error_file))


    def test_prepare_result_logging(self):
        # test that results object is given and entails hyperpipe infos
        rfc = PipelineElement('RandomForestClassifier')
        lsvc = PipelineElement('LinearSVC')
        branch = Branch('dummy_branch')
        branch += PipelineElement('SVC')
        self.hyperpipe += Stack('final_stack', [PipelineElement('SVC'), rfc, branch])
        self.hyperpipe += lsvc

        self.hyperpipe.data.X = self.__X
        self.hyperpipe.data.y = self.__y
        self.hyperpipe._prepare_result_logging(datetime.datetime.now())
        self.assertTrue(isinstance(self.hyperpipe.results, MDBHyperpipe))
        self.assertTrue(isinstance(self.hyperpipe.results_handler, ResultsHandler))
        self.assertTrue(len(self.hyperpipe.results.outer_folds) == 0)

        expected_pipeline_struct = {'StandardScaler': str(type(self.ss_pipe_element.base_element)),
                                    'PCA': str(type(self.pca_pipe_element.base_element)),
                                    'SVC': str(type(self.svc_pipe_element.base_element)),
                                    'STACK:final_stack': {'SVC': str(type(self.svc_pipe_element.base_element)),
                                                          'RandomForestClassifier': str(type(rfc.base_element)),
                                                          'BRANCH:dummy_branch': {
                                                              'SVC': str(type(self.svc_pipe_element.base_element))
                                                          }},
                                    'LinearSVC': str(type(lsvc.base_element))
                                    }
        self.assertDictEqual(self.hyperpipe.results.hyperpipe_info.elements, expected_pipeline_struct)

    def test_finalize_optimization(self):
        # it is kind of difficult to test that's why we fake it
        self.hyperpipe.fit(self.__X, self.__y)

        # reset all infos
        self.hyperpipe.results.dummy_estimator.metrics_train = MDBScoreInformation()
        self.hyperpipe.results.dummy_estimator.metrics_test = MDBScoreInformation()
        self.hyperpipe.results.metrics_train = {}
        self.hyperpipe.results.metrics_test = {}
        self.hyperpipe.best_config = None
        self.hyperpipe.results.best_config = MDBConfig()
        self.hyperpipe.optimum_pipe = None

        # now generate infos again
        self.hyperpipe._finalize_optimization()

        expected_num_of_metrics = len(self.hyperpipe.optimization.metrics)
        # dummy average values
        self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_train), expected_num_of_metrics)
        self.assertTrue(len(self.hyperpipe.results.dummy_estimator.metrics_test), expected_num_of_metrics)
        # overall average values
        self.assertTrue(len(self.hyperpipe.results.metrics_train), 2 * expected_num_of_metrics)
        self.assertTrue(len(self.hyperpipe.results.metrics_test), 2 * expected_num_of_metrics)
        # find best config
        self.assertIsNotNone(self.hyperpipe.best_config)
        self.assertIsNotNone(self.hyperpipe.results.best_config)
        self.assertEqual(self.hyperpipe.best_config, self.hyperpipe.results.best_config.config_dict)
        # set optimum pipe and params, # todo: test add preprocessing
        self.assertIsNotNone(self.hyperpipe.optimum_pipe)
        self.assertEqual(self.hyperpipe.optimum_pipe.named_steps["SVC"].base_element.C, self.hyperpipe.best_config["SVC__C"])
        # save optimum model
        self.assertTrue(os.path.isfile(os.path.join(self.hyperpipe.output_settings.results_folder,
                                                    'photon_best_model.photon')))

        # backmapping
        # because the pca is test disabled, we expect the number of features
        self.assertEqual(len(self.hyperpipe.results.best_config_feature_importances[0]), self.__X.shape[1])
        backmapped_feature_importances = os.path.join(self.hyperpipe.output_settings.results_folder,
                                                      'optimum_pipe_feature_importances_backmapped.csv')
        self.assertTrue(os.path.isfile(backmapped_feature_importances))
        loaded_array = np.loadtxt(open(backmapped_feature_importances, 'rb'), delimiter=",")
        self.assertEqual(loaded_array.shape[0], self.__X.shape[1])

    def test_finalize_optimization_preprocessing(self):
        self.hyperpipe.elements = list()

        pre_proc = Preprocessing()
        pre_proc += PipelineElement('StandardScaler')
        self.hyperpipe.add(pre_proc)
        self.hyperpipe.add(PipelineElement('SVC'))
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(os.path.isfile(os.path.join(self.hyperpipe.output_settings.results_folder,
                                                    'photon_best_model.photon')))

    def test_finalize_optimization_preprocessing_with_client(self):
        self.hyperpipe.elements = list()

        pb = ParallelBranch(name="ParallelBranch", nr_of_processes=2)
        pb += PipelineElement('LabelEncoder')
        pre_proc = Preprocessing()
        pre_proc += pb
        self.hyperpipe.add(pre_proc)
        self.hyperpipe.add(PipelineElement('SVC'))
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(os.path.isfile(os.path.join(self.hyperpipe.output_settings.results_folder,
                                                    'photon_best_model.photon')))

    def test_optimum_pipe_predict_and_predict_proba_and_transform(self):
        # find best config and test against sklearn
        self.hyperpipe.elements[-1] = PipelineElement('RandomForestClassifier', {'n_estimators': IntegerRange(4, 20,
                                                                                                              step=2)},
                                                      random_state=42)
        self.hyperpipe.fit(self.__X, self.__y)

        # the best config is without PCA so we test it

        best_config_copy = dict(self.hyperpipe.best_config)
        del best_config_copy["PCA__disabled"]
        if self.hyperpipe.best_config["PCA__disabled"]:
            sk_elements = [('StandardScaler', StandardScaler()),
                           ('RandomForestClassifier', RandomForestClassifier(random_state=42))]
        else:
            sk_elements = [('StandardScaler', StandardScaler()),
                           ('PCA', PCA(random_state=42)),
                           ('RandomForestClassifier', RandomForestClassifier(random_state=42))]
        self.sklearn_pipe = SKLPipeline(sk_elements)
        self.sklearn_pipe.set_params(**best_config_copy)
        self.sklearn_pipe.fit(self.__X, self.__y)

        self.assertTrue(np.array_equal(self.sklearn_pipe.predict(self.__X), self.hyperpipe.predict(self.__X)))
        self.assertTrue(np.array_equal(self.sklearn_pipe.predict_proba(self.__X),
                                       self.hyperpipe.predict_proba(self.__X)))
        # fake transform on sklearn pipe
        step1 = self.sklearn_pipe.named_steps["StandardScaler"].transform(self.__X)
        if "PCA" in self.sklearn_pipe.named_steps:
            step2 = self.sklearn_pipe.named_steps["PCA"].transform(self.__X)
        else:
            step2 = step1
        self.assertTrue(np.allclose(step2, self.hyperpipe.transform(self.__X)))


class HyperpipeOptimizationClassTests(unittest.TestCase):

    def test_best_config_metric(self):
        my_pipe_optimizer = Optimization('grid_search', {}, [], 'balanced_accuracy', None)
        self.assertTrue(my_pipe_optimizer.maximize_metric)
        my_pipe_optimizer = Optimization('grid_search', {}, [], 'mean_squared_error', None)
        self.assertFalse(my_pipe_optimizer.maximize_metric)

    def test_optmizer_input_str(self):
        with self.assertRaises(ValueError):
            my_pipe_optimizer = Optimization('unknown_optimizer', {}, [], 'accuracy', None)

        for name, opt_class in Optimization.OPTIMIZER_DICTIONARY.items():
            def get_optimizer(name, params={}):
                my_pipe_optimizer = Optimization(name, params, [], 'accuracy', None)
                return my_pipe_optimizer.get_optimizer()

            if name == 'smac':
                try:
                    import smac
                except ModuleNotFoundError:
                    with self.assertRaises(ModuleNotFoundError):
                        get_optimizer(name)
            if name =='switch':
                get_optimizer(name, {'name': 'random_grid_search'})
            else:
                self.assertIsInstance(get_optimizer(name), opt_class)

    def test_get_optimum_config(self):
        my_pipe_optimizer = Optimization('grid_search', {}, [], 'balanced_accuracy', None)
        list_of_tested_configs = list()
        metric_default = MDBFoldMetric(metric_name='balanced_accuracy',
                                       operation="mean",
                                       value=0.5)
        metric_best = MDBFoldMetric(metric_name='balanced_accuracy',
                                    operation="mean",
                                    value=0.99)
        metric_filter_best = MDBFoldMetric(metric_name='another_accuracy',
                                           operation="mean",
                                           value=0.75)
        metric_filter_default = MDBFoldMetric(metric_name='another_accuracy',
                                              operation="mean",
                                              value=0.55)
        # we add looser configs, one good config, and one good config that failed
        # and check if the good non-failing config is chosen
        for i in range(10):
            config = MDBConfig()
            # number 5 is the winner
            if i == 5 or i == 8:
                config.metrics_test = [metric_best, metric_filter_default]
                config.config_dict = {'any_key': 'pressed'}
            elif i == 7:
                config.metrics_test = [metric_default, metric_filter_best]
                config.config_dict = {'any_key': 'pressed'}
            else:
                config.metrics_test = [metric_default, metric_filter_default]
                config.config_dict = {'any_key': 'not_pressed'}
            if i == 8:
                config.config_failed = True
            list_of_tested_configs.append(config)

        outer_fold = MDBOuterFold()
        outer_fold.tested_config_list = list_of_tested_configs
        winner_config = outer_fold.get_optimum_config(metric=my_pipe_optimizer.best_config_metric,
                                                      maximize_metric=my_pipe_optimizer.maximize_metric)
        self.assertIs(winner_config, list_of_tested_configs[5])
        self.assertEqual(winner_config.get_test_metric('balanced_accuracy'), 0.99)

        winner_config_filtered = outer_fold.get_optimum_config(metric='another_accuracy', maximize_metric=True,
                                                               dict_filter=('any_key', 'pressed'))
        self.assertEqual(winner_config_filtered.get_test_metric('another_accuracy'), 0.75)

    def test_get_optimum_config_outer_folds(self):
        my_pipe_optimizer = Optimization('grid_search', {}, [], 'balanced_accuracy', None)

        outer_fold_list = list()
        for i in range(10):
            outer_fold = MDBOuterFold()
            outer_fold.best_config = MDBConfig()
            outer_fold.best_config.best_config_score = MDBInnerFold()
            outer_fold.best_config.best_config_score.validation = MDBScoreInformation()
            # again fold 5 wins
            if i == 5:
                outer_fold.best_config.best_config_score.validation.metrics = {'balanced_accuracy': 0.99}
            else:
                outer_fold.best_config.best_config_score.validation.metrics = {'balanced_accuracy': 0.5}
            outer_fold_list.append(outer_fold)

        best_config_outer_folds = my_pipe_optimizer.get_optimum_config_outer_folds(outer_fold_list)
        self.assertEqual(best_config_outer_folds.best_config_score.validation.metrics['balanced_accuracy'], 0.99)
        self.assertIs(best_config_outer_folds, outer_fold_list[5].best_config)
