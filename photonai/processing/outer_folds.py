import datetime
import numpy as np

from photonai.processing.inner_folds import InnerFoldManager
from photonai.processing.results_structure import MDBHelper, FoldOperations, MDBInnerFold, MDBScoreInformation
from photonai.processing.photon_folds import FoldInfo
from photonai.optimization import DummyPerformance
from photonai.photonlogger import Logger

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class OuterFoldManager:

    def __init__(self, copy_pipe_fnc,
                 optimization_info,
                 outer_fold_id,
                 cross_validation_info,
                 save_predictions: bool=False,
                 save_feature_importances: bool=False,
                 save_best_config_predictions: bool=True,
                 save_best_config_feature_importances: bool=True,
                 cache_folder=None,
                 cache_updater=None):
        # Information from the Hyperpipe about the design choices
        self.outer_fold_id = outer_fold_id
        self.cross_validaton_info = cross_validation_info
        self.optimization_info = optimization_info
        self.copy_pipe_fnc = copy_pipe_fnc

        self.save_predictions = save_predictions
        self.save_feature_importances = save_feature_importances
        self.save_best_config_predictions = save_best_config_predictions
        self.save_best_config_feature_importances = save_best_config_feature_importances
        self.cache_folder = cache_folder
        self.cache_updater = cache_updater

        # Information about the optimization progress
        self.current_best_config = None
        self.optimizer = None
        self.dummy_results = None
        self.constraint_objects = None

        # data
        self.inner_folds = None
        self.result_object = None
        self._validation_X = None
        self._validation_y = None
        self._validation_kwargs = None
        self._validation_group = None
        self._test_X = None
        self._test_y = None
        self._test_kwargs = None

    # How to get optimizer instance?

    def prepare_optimization(self, pipeline_elements: list, outer_fold_result_obj):
        self.optimizer = self.optimization_info.get_optimizer()
        # Todo: copy performance constraints for each outer fold
        self.optimizer.prepare(pipeline_elements, self.optimization_info.maximize_metric)

        # todo: we've got some super strange pymodm problems here
        #  somehow some information from the previous outer fold lingers on and can be found within a completely new
        #  instantiated OuterFoldMDB object
        #  hence, clearing it
        outer_fold_result_obj.tested_config_list = list()
        self.result_object = outer_fold_result_obj

        # copy constraint objects.
        if self.optimization_info.inner_cv_callback_functions is not None:
            self.constraint_objects = [original.copy_me() for original in self.optimization_info.inner_cv_callback_functions]
        else:
            self.constraint_objects = None

    def _prepare_data(self, X, y=None, groups=None, **kwargs):
        # Prepare Train and validation set data
        self._validation_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
        self._validation_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
        if groups is not None and len(groups) > 0:
            self._validation_group = groups[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
        self._test_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]
        self._test_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]

        # iterate over all kwargs list to prepare them for cross validation
        self._validation_kwargs = {}
        self._test_kwargs = {}
        if len(kwargs) > 0:
            for name, list_item in kwargs.items():
                if isinstance(list_item, (list, np.ndarray)):
                    self._validation_kwargs[name] = list_item[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
                    self._test_kwargs[name] = list_item[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]

        # write numbers to database info object
        self.result_object.number_samples_validation = self._validation_y.shape[0]
        self.result_object.number_samples_test = self._test_y.shape[0]
        if self.optimization_info.maximize_metric:
            self.result_object.class_distribution_validation = FoldInfo._data_overview(self._validation_y)
            self.result_object.class_distribution_test = FoldInfo._data_overview(self._test_y)

    def _generate_inner_folds(self):

        self.inner_folds = FoldInfo.generate_folds(self.cross_validaton_info.inner_cv,
                                                   self._validation_X,
                                                   self._validation_y,
                                                   self._validation_group)

        self.cross_validaton_info.inner_folds[self.outer_fold_id] = {f.fold_id: f for f in self.inner_folds}

    def fit(self, X, y=None, groups=None, **kwargs):

        self._prepare_data(X, y, groups, **kwargs)
        self._generate_inner_folds()

        outer_fold_fit_start_time = datetime.datetime.now()
        best_metric_yet = None
        tested_config_counter = 0

        # distribute number of folds to encapsulated child hyperpipes
        # self.__distribute_cv_info_to_hyperpipe_children(num_of_folds=num_folds,
        #                                                 outer_fold_counter=outer_fold_counter)

        # do the optimizing
        for current_config in self.optimizer.ask:

            tested_config_counter += 1

            if hasattr(self.optimizer, 'ask_for_pipe'):
                pipe_ctor = self.optimizer.ask_for_pipe()
            else:
                pipe_ctor = self.copy_pipe_fnc

            # self.__distribute_cv_info_to_hyperpipe_children(reset=True, config_counter=tested_config_counter)

            hp = InnerFoldManager(pipe_ctor, current_config,
                                  self.optimization_info,
                                  self.cross_validaton_info, self.outer_fold_id, self.constraint_objects,
                                  save_predictions=self.save_predictions,
                                  save_feature_importances=self.save_feature_importances,
                                  cache_folder=self.cache_folder,
                                  cache_updater=self.cache_updater)

            # Test the configuration cross validated by inner_cv object
            current_config_mdb = hp.fit(self._validation_X, self._validation_y, **self._validation_kwargs)
            current_config_mdb.config_nr = tested_config_counter

            if not current_config_mdb.config_failed:
                # get optimizer_metric and forward to optimizer
                # todo: also pass greater_is_better=True/False to optimizer
                if self.cross_validaton_info.calculate_metrics_per_fold:
                    fold_operation = FoldOperations.MEAN
                else:
                    fold_operation = FoldOperations.RAW
                metric_train = MDBHelper.get_metric(current_config_mdb, fold_operation,
                                                    self.optimization_info.best_config_metric)
                metric_test = MDBHelper.get_metric(current_config_mdb, fold_operation,
                                                   self.optimization_info.best_config_metric, train=False)

                if metric_train is None or metric_test is None:
                    raise Exception("Config did not fail, but did not get any metrics either....!!?")
                config_performance = (metric_train, metric_test)
                if best_metric_yet is None:
                    best_metric_yet = config_performance
                else:
                    # check if we have the next superstar around that exceeds any old performance
                    if self.optimization_info.maximize_metric:
                        if metric_test > best_metric_yet[1]:
                            best_metric_yet = config_performance
                    else:
                        if metric_test < best_metric_yet[1]:
                            best_metric_yet = config_performance

                # Print Result for config
                Logger().debug('...done:')
                Logger().verbose(self.optimization_info.best_config_metric + str(config_performance))
                Logger().verbose('best config performance so far: ' + str(best_metric_yet))
            else:
                config_performance = (-1, -1)
                # Print Result for config
                Logger().debug('...failed:')
                Logger().error(current_config_mdb.config_error)

            # add config to result tree and do intermediate saving
            self.result_object.tested_config_list.append(current_config_mdb)

            # 3. inform optimizer about performance
            self.optimizer.tell(current_config, config_performance)

        # generate outer cvs
        if tested_config_counter > 0:
            best_config_outer_fold = self.optimization_info.get_optimum_config(self.result_object.tested_config_list)

            if not best_config_outer_fold:
                raise Exception("No best config was found!")

            # ... and create optimal pipeline
            optimum_pipe = self.copy_pipe_fnc()
            self.cache_updater(optimum_pipe, self.cache_folder, "fixed_fold_id")
            optimum_pipe.caching = False
            # set self to best config
            optimum_pipe.set_params(**best_config_outer_fold.config_dict)

            # Todo: set all children to best config and inform to NOT optimize again, ONLY fit
            # for child_name, child_config in best_config_outer_fold_mdb.children_config_dict.items():
            #     if child_config:
            #         # in case we have a pipeline stacking we need to identify the particular subhyperpipe
            #         splitted_name = child_name.split('__')
            #         if len(splitted_name) > 1:
            #             stacking_element = self.optimum_pipe.named_steps[splitted_name[0]]
            #             pipe_element = stacking_element.elements[splitted_name[1]]
            #         else:
            #             pipe_element = self.optimum_pipe.named_steps[child_name]
            #         pipe_element.set_params(**child_config)
            #         pipe_element.is_final_fit = True

            # self.__distribute_cv_info_to_hyperpipe_children(reset=True)

            Logger().verbose('...now fitting with optimum configuration')
            optimum_pipe.fit(self._validation_X, self._validation_y, **self._validation_kwargs)

            self.result_object.best_config = best_config_outer_fold

            # save test performance
            best_config_performance_mdb = MDBInnerFold()
            best_config_performance_mdb.fold_nr = -99
            best_config_performance_mdb.number_samples_training = self._validation_y.shape[0]
            best_config_performance_mdb.number_samples_validation = self._test_y.shape[0]

            if self.cross_validaton_info.eval_final_performance:
                # Todo: generate mean and std over outer folds as well. move this items to the top
                Logger().verbose('...now predicting unseen data on test set')

                test_score_mdb = InnerFoldManager.score(optimum_pipe, self._test_X, self._test_y,
                                                        indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices,
                                                        metrics=self.optimization_info.metrics,
                                                        save_predictions=self.save_best_config_predictions,
                                                        save_feature_importances=self.save_best_config_feature_importances,
                                                        **self._test_kwargs)

                Logger().info('.. calculating metrics for test set')
                Logger().verbose('...now predicting final model with training data')

                train_score_mdb = InnerFoldManager.score(optimum_pipe, self._validation_X, self._validation_y,
                                                         indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices,
                                                         metrics=self.optimization_info.metrics,
                                                         save_predictions=self.save_best_config_predictions,
                                                         save_feature_importances=self.save_best_config_feature_importances,
                                                         training=True,
                                                         **self._validation_kwargs)

                best_config_performance_mdb.training = train_score_mdb
                best_config_performance_mdb.validation = test_score_mdb

                Logger().info('PERFORMANCE TRAIN:')
                for m_key, m_value in train_score_mdb.metrics.items():
                    Logger().info(str(m_key) + ": " + str(m_value))

                Logger().info('PERFORMANCE TEST:')
                for m_key, m_value in test_score_mdb.metrics.items():
                    Logger().info(str(m_key) + ": " + str(m_value))
            else:

                def _copy_inner_fold_means(metric_dict):
                    # We copy all mean values from validation to the best config
                    # training
                    train_item_metrics = {}
                    for m in metric_dict:
                        if m.operation == str(FoldOperations.MEAN):
                            train_item_metrics[m.metric_name] = m.value
                    train_item = MDBScoreInformation()
                    train_item.metrics_copied_from_inner = True
                    train_item.metrics = train_item_metrics
                    return train_item

                # training
                best_config_performance_mdb.training = _copy_inner_fold_means(best_config_outer_fold.metrics_train)
                # validation
                best_config_performance_mdb.validation = _copy_inner_fold_means(best_config_outer_fold.metrics_test)

            # write best config performance to best config item
            self.result_object.best_config.best_config_score = best_config_performance_mdb

        Logger().info('This took {} minutes.'.format((datetime.datetime.now() - outer_fold_fit_start_time).total_seconds() / 60))

    def fit_dummy(self, X, y, dummy):

        try:
            train_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
            train_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]

            if isinstance(train_X, np.ndarray):
                if len(train_X.shape) > 2:
                    Logger().info("Skipping dummy estimator because of too much dimensions")

            dummy.fit(train_X, train_y)
            train_scores = InnerFoldManager.score(dummy, train_X, train_y, metrics=self.optimization_info.metrics)

            # fill result tree with fold information
            inner_fold = MDBInnerFold()
            inner_fold.training = train_scores

            if self.cross_validaton_info.eval_final_performance:
                test_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]
                test_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]
                test_scores = InnerFoldManager.score(dummy, test_X, test_y, metrics=self.optimization_info.metrics)
                Logger().info("Dummy Results: " + str(test_scores))
                inner_fold.validation = test_scores

            self.dummy_results = inner_fold

            # performaceConstraints: DummyEstimator
            if self.constraint_objects is not None:
                dummy_constraint_objs = [opt for opt in self.constraint_objects if isinstance(opt, DummyPerformance)]
                if dummy_constraint_objs:
                    for dummy_constraint_obj in dummy_constraint_objs:
                        dummy_constraint_obj.set_dummy_performance(self.dummy_results)

            return inner_fold
        except Exception as e:
            Logger().error(e)
            Logger().info("Skipping dummy because of error..")
            return None

    @staticmethod
    def extract_feature_importances(optimum_pipe):
        return InnerFoldManager.extract_feature_importances(optimum_pipe)
