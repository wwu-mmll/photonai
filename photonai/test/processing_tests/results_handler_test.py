import os
import shutil
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, OutputSettings
from photonai.base import PipelineElement
from photonai.processing import ResultsHandler
from photonai.test.photon_base_test import PhotonBaseTest
from photonai.processing.results_structure import MDBHyperpipe


class ResultsHandlerTest(PhotonBaseTest):

    def setUp(self):
        """
        Set default start settings for all tests.
        """
        super(ResultsHandlerTest, self).setUp()

        self.files = ['best_config_predictions.csv',
                      'time_monitor.csv',
                      'time_monitor_pie.png',
                      'photon_result_file.json',
                      'photon_summary.txt',
                      'photon_best_model.photon',
                      'optimizer_history.png']
                    # todo: 'optimum_pipe_feature_importances_backmapped.npz',
                    # 'optimum_pipe_feature_importances_backmapped.npz',

        self.output_settings = OutputSettings(project_folder=self.tmp_folder_path, save_output=True)

        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, random_state=42)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1], 'kernel': ['linear']},  # 'rbf', 'sigmoid']
                                                random_state=42)

        self.inner_cv_object = KFold(n_splits=3)
        self.metrics = ["accuracy", 'recall', 'precision']
        self.best_config_metric = "accuracy"
        self.hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
                                   metrics=self.metrics,
                                   best_config_metric=self.best_config_metric,
                                   outer_cv=KFold(n_splits=2),
                                   output_settings=self.output_settings,
                                   verbosity=1)
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

        self.hyperpipe.fit(self.__X, self.__y)

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
        with open(os.path.join(self.output_settings.results_folder, 'photon_summary.txt')) as file:
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
                                           table[result_metric.operation.split(".")[1]][result_metric.metric_name], 4)

        splitted_areas = {}
        for num in range(len(key_areas_outer_fold)):
            splitted_areas[key_areas_outer_fold[num]] = areas[len(key_areas)+1+num*2].split("\n")

        # check all outer_folds
        for key_area_outer_fold in key_areas_outer_fold:
            if [perf for perf in splitted_areas[key_area_outer_fold] if perf == "PERFORMANCE:"]:
                index_dict[key_area_outer_fold+"_train"] = splitted_areas[key_area_outer_fold].index("PERFORMANCE:")
                index_dict[key_area_outer_fold + "_test"] = index_dict[key_area_outer_fold+"_train"]
            else:
                self.assertTrue(False)
            for data_key in [k for k in list(result_dict.keys()) if key_area_outer_fold in k]:
                table_str = "\n".join([splitted_areas[key_area_outer_fold][index_dict[data_key] + i]
                                       for i in [2, 4, 5, 6]])
                table = pd.read_csv(StringIO(table_str.replace(" ", "")), sep="|")[
                        ["MetricName", "TrainValue", "TestValue"]].set_index("MetricName")

                for result_metric in result_dict[data_key].metrics.keys():
                    self.assertAlmostEqual(result_dict[data_key].metrics[result_metric],
                                           table[outer_fold_traintest[data_key]][result_metric], 4)

    def test_save_backmapping(self):
        """
        Check dimension of feature backmapping equals input dimensions.
        """
        npzfile = np.load(os.path.join(self.output_settings.results_folder,
                                       'optimum_pipe_feature_importances_backmapped.npz'))

        self.assertEqual(len(npzfile.files), 1)
        result_data = []
        for file in npzfile.files:
            result_data.append(npzfile[file])

        self.assertEqual(np.shape(self.__X)[1], result_data[0].size)

    #  def test_save_backmapping_stack(self):
    #    self.hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
    #                               metrics=self.metrics,
    #                               best_config_metric=self.best_config_metric,
    #                               outer_cv=KFold(n_splits=2),
    #                               output_settings=self.output_settings,
    #                               verbosity=1)
    #    self.hyperpipe += self.ss_pipe_element
    #    self.stack = Stack("myStack")
    #    self.stack += PipelineElement("MinMaxScaler")
    #    self.stack += self.pca_pipe_element
    #    self.hyperpipe += self.stack
    #    self.hyperpipe.add(self.svc_pipe_element)

    #    self.output_settings.save_output = True
    #    self.hyperpipe.fit(self.__X, self.__y)
    #    picklefile = pickle.load(open(
    #        os.path.join(self.output_settings.results_folder, 'optimum_pipe_feature_importances_backmapped.p'),"rb"))

    #    self.assertEqual(np.shape(self.__X)[1], len(picklefile[0]))

    def pass_through_plots(self):
        """
        Test for plot functions. Only passing test, no quality testing.
        """
        self.assertIsNone(self.hyperpipe.results.plot_optimizer_history())
        self.assertIsNone(self.hyperpipe.results.plot_true_pred())
        self.assertIsNone(self.hyperpipe.results.plot_confusion_matrix())
        self.assertIsNone(self.hyperpipe.results.plot_roc_curve())

    def test_save_all_learning_curves(self):
        """
        Test number of saved learning curve files
        """
        hyperpipe = Hyperpipe('god', inner_cv=self.inner_cv_object,
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

    def test_load_from_file(self):
        X, y = load_breast_cancer(True)
        my_pipe = Hyperpipe('load_results_file_test',
                            inner_cv=KFold(n_splits=3),
                            metrics=['accuracy'],
                            best_config_metric='accuracy',
                            output_settings=OutputSettings(project_folder='./tmp'))
        my_pipe += PipelineElement("StandardScaler")
        my_pipe += PipelineElement("SVC")
        my_pipe.fit(X, y)

        results_file = os.path.join(my_pipe.output_settings.results_folder, "photon_result_file.json")
        my_result_handler = ResultsHandler()
        my_result_handler.load_from_file(results_file)
        self.assertIsInstance(my_result_handler.results, MDBHyperpipe)

    def test_get_performance_table(self):
        pass

    def test_load_from_mongodb(self):
        pass
