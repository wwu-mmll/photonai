import os
import shutil
import uuid
import warnings
from prettytable import PrettyTable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

from photonai.base import Hyperpipe, OutputSettings
from photonai.base import PipelineElement, Stack, Switch
from photonai.optimization import IntegerRange, FloatRange
from photonai.processing import ResultsHandler
from photonai.helper.photon_base_test import PhotonBaseTest
from photonai.processing.results_structure import MDBHyperpipe


class ResultsHandlerTest(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(ResultsHandlerTest, cls).setUpClass()

        cls.files = ['best_config_predictions.csv',
                     'photon_result_file.json',
                     'photon_summary.txt',
                     'photon_best_model.photon']

        cls.results_folder = cls.tmp_folder_path
        cls.output_settings = OutputSettings(save_output=True)

        cls.mongodb_path = 'mongodb://localhost:27017/photon_results'

        cls.ss_pipe_element = PipelineElement('StandardScaler')
        cls.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, random_state=42)
        cls.svc_pipe_element = PipelineElement('SVC', {'C': [0.1], 'kernel': ['linear']},  # 'rbf', 'sigmoid']
                                               random_state=42)

        cls.inner_cv_object = KFold(n_splits=3)
        cls.metrics = ["accuracy", 'recall', 'precision']
        cls.best_config_metric = "accuracy"
        cls.pipe_name = str(uuid.uuid4())
        cls.hyperpipe = Hyperpipe(cls.pipe_name, inner_cv=cls.inner_cv_object,
                                  metrics=cls.metrics,
                                  best_config_metric=cls.best_config_metric,
                                  outer_cv=KFold(n_splits=2),
                                  project_folder=cls.results_folder,
                                  output_settings=cls.output_settings,
                                  verbosity=0)
        cls.hyperpipe += cls.ss_pipe_element
        cls.hyperpipe += cls.pca_pipe_element
        cls.hyperpipe.add(cls.svc_pipe_element)

        dataset = load_breast_cancer()
        cls.__X = dataset.data
        cls.__y = dataset.target

        cls.hyperpipe.fit(cls.__X, cls.__y)
        
    def test_write_convenience_files(self):
        """
        Output creation testing. Only write if output_settings.save_output == True
        """
        for file in self.files:
            self.assertTrue(os.path.isfile(os.path.join(self.hyperpipe.output_settings.results_folder, file)))

        # correct rows
        with open(os.path.join(self.hyperpipe.output_settings.results_folder, 'best_config_predictions.csv')) as f:
            self.assertEqual(sum([outer_fold.number_samples_test
                                  for outer_fold in self.hyperpipe.results.outer_folds]),
                             sum(1 for _ in f)-1)

        shutil.rmtree(self.tmp_folder_path, ignore_errors=True)
        self.hyperpipe.output_settings = OutputSettings(save_output=False)
        self.hyperpipe.fit(self.__X, self.__y)
        self.assertFalse(os.path.exists(self.hyperpipe.output_settings.results_folder))

    def test_summary(self):
        """
        Check content of photon_summary.txt. Adjustment with hyperpipe.result.
        """

        # todo: check appropriately
        pass

    def test_save_backmapping_weird_format(self):
        self.hyperpipe.fit(self.__X, self.__y)
        weird_format_fake = ('string', ['array'])
        # should be saved as pickle
        self.hyperpipe.results_handler.save_backmapping('weird_format', weird_format_fake)
        expected_file = os.path.join(self.hyperpipe.output_settings.results_folder, 'weird_format.p')
        self.assertTrue(os.path.isfile(expected_file))

    def test_save_backmapping_csv(self):
        """
        Check dimension of feature backmapping equals input dimensions for less than 1000 features.
        """
        backmapping = np.loadtxt(os.path.join(self.hyperpipe.output_settings.results_folder,
                                 'optimum_pipe_feature_importances_backmapped.csv'), delimiter=',')
        self.assertEqual(np.shape(self.__X)[1], backmapping.size)

    def test_save_backmapping_npz(self):
        """
        Check dimension of feature backmapping equals input dimensions for more than 1000 features.
        """
        # run another hyperpipe with more than 1000 features
        # use np.tile to copy features until at least 1000 features are reached
        X = np.tile(self.__X, (1, 35))
        self.hyperpipe.fit(X, self.__y)
        npzfile = np.load(os.path.join(self.hyperpipe.output_settings.results_folder,
                                       'optimum_pipe_feature_importances_backmapped.npz'))
        self.assertEqual(len(npzfile.files), 1)
        backmapping = npzfile[npzfile.files[0]]
        self.assertEqual(np.shape(X)[1], backmapping.size)

    def test_save_backmapping_stack(self):
        # build hyperpipe with stack first
        self.hyperpipe = Hyperpipe(self.pipe_name, inner_cv=self.inner_cv_object,
                                   metrics=self.metrics,
                                   best_config_metric=self.best_config_metric,
                                   outer_cv=KFold(n_splits=2),
                                   output_settings=self.output_settings,
                                   project_folder=self.results_folder,
                                   verbosity=1)
        self.hyperpipe += self.ss_pipe_element
        self.stack = Stack("myStack")
        self.stack += PipelineElement("MinMaxScaler")
        self.stack += self.pca_pipe_element
        self.hyperpipe += self.stack
        self.hyperpipe.add(self.svc_pipe_element)

        self.output_settings.save_output = True
        self.hyperpipe.fit(self.__X, self.__y)

        f_name = os.path.join(self.hyperpipe.output_settings.results_folder,
                              'optimum_pipe_feature_importances_backmapped.csv')
        self.assertTrue(not os.path.isfile(f_name))

    def test_save_backmapping_reduced_dimension_without_inverse(self):
        class RD(BaseEstimator, TransformerMixin):

            def fit(self, X, y, **kwargs):
                pass

            def fit_transform(self, X, y=None, **fit_params):
                return self.transform(X)

            def transform(self, X):
                return X[:, :3]

        trans = PipelineElement.create('MyTransformer', base_element=RD(), hyperparameters={})

        self.hyperpipe = Hyperpipe(self.pipe_name, inner_cv=self.inner_cv_object,
                                   metrics=self.metrics,
                                   best_config_metric=self.best_config_metric,
                                   outer_cv=KFold(n_splits=2),
                                   project_folder=self.results_folder,
                                   output_settings=self.output_settings,
                                   verbosity=1)
        self.hyperpipe += trans
        self.hyperpipe.add(self.svc_pipe_element)

        self.output_settings.save_output = True

        f_name = os.path.join(self.results_folder, 'optimum_pipe_feature_importances_backmapped.csv')
        self.assertTrue(not os.path.isfile(f_name))

    def test_get_feature_importances(self):
        self.hyperpipe.elements[-1] = PipelineElement('LinearSVC')
        self.hyperpipe.fit(self.__X, self.__y)
        f_imps = self.hyperpipe.results_handler.get_importance_scores()
        self.assertIsNotNone(f_imps)

    def test_pass_through_plots(self):
        """
        Test for plot functions. Only passing test, no quality testing.
        """
        self.assertIsNone(self.hyperpipe.results_handler.plot_optimizer_history(metric='accuracy', type='scatter'))
        self.assertIsNone(self.hyperpipe.results_handler.eval_mean_time_components())

    def test_empty_results(self):
        res_handler = ResultsHandler()
        with self.assertRaises(ValueError):
            res_handler.get_test_predictions('test_predictions')
        with self.assertRaises(ValueError):
            res_handler.get_validation_predictions('test_predictions')

    def test_save_all_learning_curves(self):
        """
        Test count of saved learning curve files.
        """
        hyperpipe = Hyperpipe(self.pipe_name, inner_cv=self.inner_cv_object,
                              learning_curves=True,
                              metrics=self.metrics,
                              best_config_metric=self.best_config_metric,
                              outer_cv=KFold(n_splits=2),
                              project_folder=self.results_folder,
                              output_settings=self.output_settings,
                              verbosity=2)
        hyperpipe += self.ss_pipe_element
        hyperpipe += self.pca_pipe_element
        hyperpipe.add(self.svc_pipe_element)
        hyperpipe.fit(self.__X, self.__y)
        results_handler = hyperpipe.results_handler
        results_handler.save_all_learning_curves()
        config_num = len(hyperpipe.results.outer_folds[0].tested_config_list)
        target_file_num = 2 * config_num * hyperpipe.cross_validation.outer_cv.n_splits
        curves_folder = hyperpipe.output_settings.results_folder + '/learning_curves/'
        true_file_num = len([name for name in os.listdir(curves_folder)
                             if os.path.isfile(os.path.join(curves_folder, name))])
        self.assertEqual(target_file_num, true_file_num)

        # test outer fold plots
        results_handler.plot_learning_curves_outer_fold(0)

    def test_load_from_file_and_mongodb(self):
        hyperpipe = self.hyperpipe.copy_me()
        hyperpipe.output_settings.mongodb_connect_url = self.mongodb_path
        hyperpipe.fit(self.__X, self.__y)

        results_file = os.path.join(hyperpipe.output_settings.results_folder, "photon_result_file.json")
        my_result_handler = ResultsHandler()
        my_result_handler.load_from_file(results_file)
        self.assertIsInstance(my_result_handler.results, MDBHyperpipe)

        my_mongo_result_handler = ResultsHandler()
        my_mongo_result_handler.load_from_mongodb(pipe_name=hyperpipe.name,
                                                  mongodb_connect_url=self.mongodb_path)
        self.assertIsInstance(my_mongo_result_handler.results, MDBHyperpipe)
        self.assertTrue(my_mongo_result_handler.results.name == hyperpipe.name)

        # write again to mongodb
        hyperpipe.fit(self.__X, self.__y)
        with warnings.catch_warnings(record=True) as w:
            my_mongo_result_handler.load_from_mongodb(pipe_name=hyperpipe.name,
                                                      mongodb_connect_url=self.mongodb_path)
            assert any("Found multiple hyperpipes with that name." in s for s in [e.message.args[0] for e in w])

        with self.assertRaises(FileNotFoundError):
            my_result_handler.load_from_mongodb(pipe_name='any_weird_name_1238934384834234892384382',
                                                mongodb_connect_url=self.mongodb_path)

    def test_get_performance_table(self):

        performance_table = self.hyperpipe.results_handler.get_performance_table()

        self.assertListEqual(list(performance_table.columns), ["best_config", "fold", "n_train", "n_validation"] +
                             self.metrics +
                             [x+"_sem" for x in self.metrics])

        for i in range(2):
            self.assertEqual(int(performance_table.iloc[i]["fold"]), i+1)
            for key, value in self.hyperpipe.results_handler.results.outer_folds[i].\
                    best_config.best_config_score.validation.metrics.items():
                self.assertEqual(performance_table.iloc[i][key], value)
            self.assertEqual(self.hyperpipe.results_handler.results.outer_folds[i].best_config.
                             best_config_score.number_samples_training,
                             performance_table.iloc[i]["n_train"])
            self.assertEqual(self.hyperpipe.results_handler.results.outer_folds[i].best_config.
                             best_config_score.number_samples_validation,
                             performance_table.iloc[i]["n_validation"])
        self.assertEqual(performance_table.iloc[2].isnull().all(), False)

        self.assertIsInstance(performance_table, pd.DataFrame)

    def test_get_methods(self):
        self.hyperpipe.results_handler.get_methods()

    def test_float_labels_with_mongo(self):
        """
        This test was added for a bug with float labels and saving to mongoDB.
        """
        local_y = self.__y.astype(float)
        self.hyperpipe.output_settings.mongodb_connect_url = self.mongodb_path
        self.hyperpipe.fit(self.__X, local_y)

    def test_get_mean_of_best_validation_configs_per_estimator(self):
        hyperpipe = Hyperpipe('compare_estimators',
                              inner_cv=KFold(n_splits=2, shuffle=True),
                              outer_cv=KFold(n_splits=2, shuffle=True),
                              metrics=['mean_squared_error'],
                              best_config_metric='mean_squared_error',
                              optimizer='switch',
                              optimizer_params={'name': 'random_search', 'n_configurations': 3},
                              project_folder='./tmp',
                              verbosity=0)
        hyperpipe += self.ss_pipe_element

        # compare different learning algorithms in an OR_Element
        estimators = Switch('estimator_selection')
        estimators += PipelineElement('RandomForestRegressor',
                                      hyperparameters={'min_samples_split': IntegerRange(2, 4)})
        estimators += PipelineElement('SVR', hyperparameters={'C': FloatRange(0.5, 25)})

        hyperpipe += estimators
        hyperpipe.fit(self.__X[:150], self.__y[:150])

        output = hyperpipe.results_handler.get_mean_of_best_validation_configs_per_estimator()
        output2 = hyperpipe.results_handler.get_n_best_validation_configs_per_estimator()

        self.assertTrue(isinstance(output, PrettyTable))
        self.assertTrue(isinstance(output2, dict))
