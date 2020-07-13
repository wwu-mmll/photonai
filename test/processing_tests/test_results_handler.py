import os
import shutil
from io import StringIO
import uuid
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, OutputSettings
from photonai.base import PipelineElement, Stack
from photonai.processing import ResultsHandler
from photonai.helper.photon_base_test import PhotonBaseTest
from photonai.processing.results_structure import MDBHyperpipe


class ResultsHandlerTest(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(ResultsHandlerTest, cls).setUpClass()

        cls.files = ['best_config_predictions.csv',
                      'time_monitor.csv',
                      'time_monitor_pie.png',
                      'photon_result_file.json',
                      'photon_summary.txt',
                      'photon_best_model.photon',
                      'optimizer_history.png']
        # todo: 'optimum_pipe_feature_importances_backmapped.npz',
        # 'optimum_pipe_feature_importances_backmapped.npz',

        cls.output_settings = OutputSettings(project_folder=cls.tmp_folder_path,
                                             save_output=True)

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
                                  output_settings=cls.output_settings,
                                  verbosity=1)
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
            self.assertTrue(os.path.isfile(os.path.join(self.output_settings.results_folder, file)))

        # correct rows
        with open(os.path.join(self.output_settings.results_folder, 'best_config_predictions.csv')) as f:
            self.assertEqual(sum([outer_fold.number_samples_test
                                  for outer_fold in self.hyperpipe.results.outer_folds]),
                             sum(1 for _ in f)-1)

        shutil.rmtree(self.tmp_folder_path, ignore_errors=True)
        self.output_settings = OutputSettings(project_folder=self.tmp_folder_path, save_output=False)
        self.hyperpipe.fit(self.__X, self.__y)
        self.assertIsNone(self.output_settings.results_folder)

    def test_readable_time_monitor_csv(self):
        """
        Test for only readable time_moitor.csv (right count of columns and pandas import).
        """
        time_monitor_df = pd.read_csv(os.path.join(self.output_settings.results_folder, 'time_monitor.csv'),
                                      header=[0, 1])
        self.assertIsInstance(time_monitor_df, pd.DataFrame)
        self.assertEqual(len(time_monitor_df.columns), 10)

    def test_summary(self):
        """
        Check content of photon_summary.txt. Adjustment with hyperpipe.result.
        """
        self.hyperpipe.fit(self.__X, self.__y)
        with open(os.path.join(self.hyperpipe.output_settings.results_folder, 'photon_summary.txt')) as file:
            data = file.read()

        areas = data.split("-------------------------------------------------------------------")

        # first areas
        self.assertEqual(areas[0], "\nPHOTON RESULT SUMMARY\n")

        result_dict = {"dummy_test": self.hyperpipe.results.dummy_estimator.test,
                       "dummy_train": self.hyperpipe.results.dummy_estimator.train,
                       "best_config_train": self.hyperpipe.results.metrics_train,
                       "best_config_test": self.hyperpipe.results.metrics_test}

        outer_fold_traintest = {}

        key_areas_outer_fold = []
        # all outerfold areas
        for i in range(len(self.hyperpipe.results.outer_folds)):
            self.assertEqual(areas[4+i*2], '\nOUTER FOLD '+str(i+1)+'\n')
            key_areas_outer_fold.append("outer_fold_"+str(i+1))
            result_dict["outer_fold_"+str(i+1)+"_train"] = \
                self.hyperpipe.results.outer_folds[i].best_config.best_config_score.training
            outer_fold_traintest["outer_fold_"+str(i+1)+"_train"] = "TrainValue"
            result_dict["outer_fold_" + str(i + 1) + "_test"] = \
                self.hyperpipe.results.outer_folds[i].best_config.best_config_score.validation
            outer_fold_traintest["outer_fold_"+str(i+1)+"_test"] = "TestValue"

        # check performance / test-train of dummy and best_config
        key_areas = ["entracee", "name", "dummy", "best_config"]
        splitted_areas = {}

        for num in range(len(key_areas)):
            splitted_areas[key_areas[num]] = areas[num].split("\n")

        index_dict = {}
        for key in key_areas[2:]:
            if [perf for perf in splitted_areas[key] if perf == "TEST:"]:
                index_dict[key+"_test"] = splitted_areas[key].index("TEST:")
                index_dict[key+"_train"] = splitted_areas[key].index("TRAINING:")
            else:
                self.assertTrue(False)
            for data_key in [k for k in list(result_dict.keys()) if key in k]:
                table_str = "\n".join([splitted_areas[key][index_dict[data_key]+i] for i in [2, 4, 5, 6]])
                table = pd.read_csv(StringIO(table_str.replace(" ", "")),
                                    sep="|")[["MetricName", "MEAN", "STD"]].set_index("MetricName")
                for result_metric in result_dict[data_key]:
                    self.assertAlmostEqual(result_metric.value,
                                           table[result_metric.operation.split(".")[1]][result_metric.metric_name], 2)

        splitted_areas = {}
        for num in range(len(key_areas_outer_fold)):
            splitted_areas[key_areas_outer_fold[num]] = areas[len(key_areas)+1+num*2].split("\n")

        # check all outer_folds
        for ka in key_areas_outer_fold:
            if [perf for perf in splitted_areas[ka] if perf == "PERFORMANCE:"]:
                index_dict[ka + "_train"] = splitted_areas[ka].index("PERFORMANCE:")
                index_dict[ka + "_test"] = index_dict[ka+"_train"]
            else:
                self.assertTrue(False)
            for data_key in [k for k in list(result_dict.keys()) if ka in k]:
                table_str = "\n".join([splitted_areas[ka][index_dict[data_key] + i]
                                       for i in [2, 4, 5, 6]])
                table = pd.read_csv(StringIO(table_str.replace(" ", "")), sep="|")[
                        ["MetricName", "TrainValue", "TestValue"]].set_index("MetricName")

                for result_metric in result_dict[data_key].metrics.keys():
                    self.assertAlmostEqual(result_dict[data_key].metrics[result_metric],
                                           table[outer_fold_traintest[data_key]][result_metric], 4)

    def test_save_backmapping_csv(self):
        """
        Check dimension of feature backmapping equals input dimensions for less than 1000 features.
        """
        backmapping = np.loadtxt(os.path.join(self.output_settings.results_folder,
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
        npzfile = np.load(os.path.join(self.output_settings.results_folder,
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
                                   verbosity=1)
        self.hyperpipe += self.ss_pipe_element
        self.stack = Stack("myStack")
        self.stack += PipelineElement("MinMaxScaler")
        self.stack += self.pca_pipe_element
        self.hyperpipe += self.stack
        self.hyperpipe.add(self.svc_pipe_element)

        self.output_settings.save_output = True
        self.hyperpipe.fit(self.__X, self.__y)
        backmapping = np.loadtxt(os.path.join(self.output_settings.results_folder,
                                              'optimum_pipe_feature_importances_backmapped.csv'), delimiter=',')

        # since backmapping stops at a stack element, we will get the features that went into the SVC, that is the
        # number of PCs from the PCA and the number of input features to the MinMaxScaler
        n_stack_features = self.hyperpipe.best_config['myStack__PCA__n_components'] + np.shape(self.__X)[1]
        self.assertEqual(n_stack_features, backmapping.size)

    def pass_through_plots(self):
        """
        Test for plot functions. Only passing test, no quality testing.
        """
        self.assertIsNone(self.hyperpipe.results.plot_optimizer_history(type='scatter'))
        self.assertIsNone(self.hyperpipe.results.plot_true_pred())
        self.assertIsNone(self.hyperpipe.results.plot_confusion_matrix())
        self.assertIsNone(self.hyperpipe.results.plot_roc_curve())

    def test_save_all_learning_curves(self):
        """
        Test number of saved learning curve files
        """
        hyperpipe = Hyperpipe(self.pipe_name, inner_cv=self.inner_cv_object,
                              learning_curves=True,
                              metrics=self.metrics,
                              best_config_metric=self.best_config_metric,
                              outer_cv=KFold(n_splits=2),
                              output_settings=self.output_settings,
                              verbosity=1)
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
        self.hyperpipe.fit(self.__X, self.__y)

        results_file = os.path.join(self.hyperpipe.output_settings.results_folder, "photon_result_file.json")
        my_result_handler = ResultsHandler()
        my_result_handler.load_from_file(results_file)
        self.assertIsInstance(my_result_handler.results, MDBHyperpipe)

        mongo_inst = my_result_handler.load_from_mongodb(pipe_name=self.pipe_name,
                                                         mongodb_connect_url=self.mongodb_path)
        self.assertIsInstance(mongo_inst, MDBHyperpipe)
        self.assertTrue(mongo_inst.name == self.pipe_name)

    def test_get_performance_table(self):
        pt = self.hyperpipe.results_handler.get_performance_table()
        self.assertIsInstance(pt, pd.DataFrame)

    def test_get_methods(self):
        self.hyperpipe.results_handler.get_methods()
