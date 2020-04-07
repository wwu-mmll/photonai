import datetime
import warnings
import numpy as np
import json

from prettytable import PrettyTable

from photonai.helper.helper import PhotonDataHelper, print_double_metrics, print_metrics
from photonai.optimization import DummyPerformance
from photonai.photonlogger.logger import logger
from photonai.processing.inner_folds import InnerFoldManager
from photonai.processing.photon_folds import FoldInfo
from photonai.processing.results_structure import MDBHelper, FoldOperations, MDBInnerFold, MDBScoreInformation
from photonai.optimization.base_optimizer import PhotonSlaveOptimizer, PhotonMasterOptimizer

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class OuterFoldManager:

    def __init__(self, pipe,
                 optimization_info,
                 outer_fold_id,
                 cross_validation_info,
                 learning_curves,
                 learning_curves_cut,
                 cache_folder=None,
                 cache_updater=None,
                 dummy_estimator=None,
                 result_obj=None):
        # Information from the Hyperpipe about the design choices
        self.outer_fold_id = outer_fold_id
        self.cross_validaton_info = cross_validation_info
        self.optimization_info = optimization_info
        self._pipe = pipe
        self.copy_pipe_fnc = self._pipe.copy_me
        self.dummy_estimator = dummy_estimator

        self.cache_folder = cache_folder
        self.cache_updater = cache_updater

        # Information about the optimization progress
        self.current_best_config = None
        self.optimizer = None
        self.constraint_objects = None

        # Information about learning curves
        self.learning_curves = learning_curves
        self.learning_curves_cut = learning_curves_cut

        # data
        self.result_object = result_obj
        self.inner_folds = None
        self._validation_X = None
        self._validation_y = None
        self._validation_kwargs = None
        self._test_X = None
        self._test_y = None
        self._test_kwargs = None

    # How to get optimizer instance?

    def _prepare_optimization(self):

        logger.info("Preparing Hyperparameter Optimization...")
        pipeline_elements = [e for name, e in self._pipe.elements]

        self.optimizer = self.optimization_info.get_optimizer()
        if isinstance(self.optimizer, PhotonMasterOptimizer):
            self.optimizer.prepare(pipeline_elements, self.optimization_info.maximize_metric, self.objectiv_function)
        else:
            self.optimizer.prepare(pipeline_elements, self.optimization_info.maximize_metric)

        # we've got some super strange pymodm problems here
        # somehow some information from the previous outer fold lingers on and can be found within a completely new
        # instantiated OuterFoldMDB object
        # hence, clearing it
        self.result_object.tested_config_list = list()

        # copy constraint objects.
        if self.optimization_info.performance_constraints is not None:
            if isinstance(self.optimization_info.performance_constraints, list):
                self.constraint_objects = [original.copy_me() for original in self.optimization_info.performance_constraints]
            else:
                self.constraint_objects = [self.optimization_info.performance_constraints.copy_me()]
        else:
            self.constraint_objects = None

    def _prepare_data(self, X, y=None, **kwargs):
        logger.info("Preparing data for outer fold " + str(self.cross_validaton_info.outer_folds[self.outer_fold_id].fold_nr) + "...")
        # Prepare Train and validation set data
        train_indices = self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices
        test_indices = self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices
        self._validation_X, self._validation_y, self._validation_kwargs = PhotonDataHelper.split_data(X, y, kwargs,
                                                                                                      indices=train_indices)
        self._test_X, self._test_y, self._test_kwargs = PhotonDataHelper.split_data(X, y, kwargs, indices=test_indices)

        # write numbers to database info object
        self.result_object.number_samples_validation = self._validation_y.shape[0]
        self.result_object.number_samples_test = self._test_y.shape[0]
        if self._pipe._estimator_type == "classifier":
            self.result_object.class_distribution_validation = FoldInfo.data_overview(self._validation_y)
            self.result_object.class_distribution_test = FoldInfo.data_overview(self._test_y)

    def _generate_inner_folds(self):

        self.inner_folds = FoldInfo.generate_folds(self.cross_validaton_info.inner_cv,
                                                   self._validation_X,
                                                   self._validation_y,
                                                   self._validation_kwargs)

        self.cross_validaton_info.inner_folds[self.outer_fold_id] = {f.fold_id: f for f in self.inner_folds}

    def fit(self, X, y=None, **kwargs):
        logger.photon_system_log('')
        logger.photon_system_log('***************************************************************************************************************')
        logger.photon_system_log('Outer Cross validation Fold {}'.format(self.cross_validaton_info.outer_folds[self.outer_fold_id].fold_nr))
        logger.photon_system_log('***************************************************************************************************************')

        self._prepare_data(X, y, **kwargs)
        self._fit_dummy()
        self._generate_inner_folds()
        self._prepare_optimization()

        outer_fold_fit_start_time = datetime.datetime.now()
        self.best_metric_yet = None
        self.tested_config_counter = 0

        # distribute number of folds to encapsulated child hyperpipes
        # self.__distribute_cv_info_to_hyperpipe_children(num_of_folds=num_folds,
        #                                                 outer_fold_counter=outer_fold_counter)

        if self.cross_validaton_info.calculate_metrics_per_fold:
            self.fold_operation = FoldOperations.MEAN
        else:
            self.fold_operation = FoldOperations.RAW

        self.max_nr_of_configs = ''
        if hasattr(self.optimizer, 'n_configurations'):
            self.max_nr_of_configs = str(self.optimizer.n_configurations)

        if isinstance(self.optimizer, PhotonMasterOptimizer):
            self.optimizer.optimize()
        else:
            # do the optimizing
            for current_config in self.optimizer.ask:
                self.objectiv_function(current_config)

        logger.clean_info('---------------------------------------------------------------------------------------------------------------')
        logger.info('Hyperparameter Optimization finished. Now finding best configuration .... ')
        print(self.tested_config_counter)
        # now go on with the best config found
        if self.tested_config_counter > 0:
            best_config_outer_fold = self.optimization_info.get_optimum_config(self.result_object.tested_config_list,
                                                                               self.fold_operation)

            if not best_config_outer_fold:
                raise Exception("No best config was found!")

            # ... and create optimal pipeline
            optimum_pipe = self.copy_pipe_fnc()
            if self.cache_updater is not None:
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

            logger.debug('Fitting model with best configuration of outer fold...')
            optimum_pipe.fit(self._validation_X, self._validation_y, **self._validation_kwargs)

            self.result_object.best_config = best_config_outer_fold

            # save test performance
            best_config_performance_mdb = MDBInnerFold()
            best_config_performance_mdb.fold_nr = -99
            best_config_performance_mdb.number_samples_training = self._validation_y.shape[0]
            best_config_performance_mdb.number_samples_validation = self._test_y.shape[0]
            best_config_performance_mdb.feature_importances = optimum_pipe.feature_importances_

            if self.cross_validaton_info.eval_final_performance:
                # Todo: generate mean and std over outer folds as well. move this items to the top
                logger.info('Calculating best model performance on test set...')

                logger.debug('...scoring test data')
                test_score_mdb = InnerFoldManager.score(optimum_pipe, self._test_X, self._test_y,
                                                        indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices,
                                                        metrics=self.optimization_info.metrics,
                                                        **self._test_kwargs)

                logger.debug('... scoring training data')

                train_score_mdb = InnerFoldManager.score(optimum_pipe, self._validation_X, self._validation_y,
                                                         indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices,
                                                         metrics=self.optimization_info.metrics,
                                                         training=True,
                                                         **self._validation_kwargs)

                best_config_performance_mdb.training = train_score_mdb
                best_config_performance_mdb.validation = test_score_mdb

                print_double_metrics(train_score_mdb.metrics, test_score_mdb.metrics)
            else:

                def _copy_inner_fold_means(metric_dict):
                    # We copy all mean values from validation to the best config
                    # training
                    train_item_metrics = {}
                    for m in metric_dict:
                        if m.operation == str(self.fold_operation):
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

        logger.info('Computations in outer fold {} took {} minutes.'.format(self.cross_validaton_info.outer_folds[self.outer_fold_id].fold_nr,
                                                                            (datetime.datetime.now() - outer_fold_fit_start_time).total_seconds() / 60))

    def objectiv_function(self, current_config):
        if isinstance(self.optimizer, PhotonMasterOptimizer):
            #current_config = [{key: x.get_dictionary()[key] for key in x.get_dictionary().keys()
            #                  if 'algos' not in key} for x in [current_config]][0]
            current_config = {k: current_config[k] for k in current_config if (current_config[k] and 'algos' not in k)}

        if current_config is None:
            return
        logger.clean_info(
            '---------------------------------------------------------------------------------------------------------------')
        self.tested_config_counter += 1

        if hasattr(self.optimizer, 'ask_for_pipe'):
            pipe_ctor = self.optimizer.ask_for_pipe()
        else:
            pipe_ctor = self.copy_pipe_fnc

        # self.__distribute_cv_info_to_hyperpipe_children(reset=True, config_counter=tested_config_counter)

        hp = InnerFoldManager(pipe_ctor, current_config,
                              self.optimization_info,
                              self.cross_validaton_info, self.outer_fold_id, self.constraint_objects,
                              cache_folder=self.cache_folder,
                              cache_updater=self.cache_updater)

        # Test the configuration cross validated by inner_cv object
        current_config_mdb = hp.fit(self._validation_X, self._validation_y, **self._validation_kwargs)
        current_config_mdb.config_nr = self.tested_config_counter

        if not current_config_mdb.config_failed:
            metric_train = MDBHelper.get_metric(current_config_mdb, self.fold_operation,
                                                self.optimization_info.best_config_metric)
            metric_test = MDBHelper.get_metric(current_config_mdb, self.fold_operation,
                                               self.optimization_info.best_config_metric, train=False)

            if metric_train is None or metric_test is None:
                raise Exception("Config did not fail, but did not get any metrics either....!!?")
            config_performance = (metric_train, metric_test)
            if self.best_metric_yet is None:
                best_metric_yet = config_performance
                self.current_best_config = current_config_mdb
            else:
                # check if we have the next superstar around that exceeds any old performance
                if self.optimization_info.maximize_metric:
                    if metric_test > self.best_metric_yet[1]:
                        best_metric_yet = config_performance
                        self.current_best_config.save_memory()
                        self.current_best_config = current_config_mdb
                    else:
                        current_config_mdb.save_memory()
                else:
                    if metric_test < self.best_metric_yet[1]:
                        best_metric_yet = config_performance
                        self.current_best_config.save_memory()
                        self.current_best_config = current_config_mdb
                    else:
                        current_config_mdb.save_memory()

            # Print Result for config
            computation_duration = current_config_mdb.computation_end_time - current_config_mdb.computation_start_time
            logger.info('Computed configuration ' + str(self.tested_config_counter) + "/" + self.max_nr_of_configs +
                        " in " + str(computation_duration))
            logger.info("Performance:             " + self.optimization_info.best_config_metric
                        + " - Train: " + "%.4f" % config_performance[0] + ", Validation: " + "%.4f" %
                        config_performance[1])
            logger.info("Best Performance So Far: " + self.optimization_info.best_config_metric
                        + " - Train: " + "%.4f" % best_metric_yet[0] + ", Validation: " + "%.4f" % best_metric_yet[1])
        else:
            config_performance = (-1, -1)
            # Print Result for config
            logger.debug('...failed:')
            logger.error(current_config_mdb.config_error)

        # add config to result tree
        self.result_object.tested_config_list.append(current_config_mdb)

        # 3. inform optimizer about performance
        logger.debug("Telling hyperparameter optimizer about recent performance.")
        if isinstance(self.optimizer, PhotonSlaveOptimizer):
            self.optimizer.tell(current_config, config_performance)
        logger.debug("Asking hyperparameter optimizer for new config.")

        if self.optimization_info.maximize_metric:
            return 1 - config_performance[1]
        else:
            return config_performance[1]

    def _fit_dummy(self):
        if self.dummy_estimator is not None:
            logger.info("Running Dummy Estimator...")
            try:
                if isinstance(self._validation_X, np.ndarray):
                    if len(self._validation_X.shape) > 2:
                        logger.info("Skipping dummy estimator because of too many dimensions")
                        self.result_object.dummy_results = None
                        return
                dummy_y = np.reshape(self._validation_y, (-1, 1))
                self.dummy_estimator.fit(dummy_y, self._validation_y)
                train_scores = InnerFoldManager.score(self.dummy_estimator, self._validation_X, self._validation_y,
                                                      metrics=self.optimization_info.metrics)

                # fill result tree with fold information
                inner_fold = MDBInnerFold()
                inner_fold.training = train_scores

                if self.cross_validaton_info.eval_final_performance:
                    test_scores = InnerFoldManager.score(self.dummy_estimator,
                                                         self._test_X, self._test_y,
                                                         metrics=self.optimization_info.metrics)
                    print_metrics("DUMMY", test_scores.metrics)
                    inner_fold.validation = test_scores

                self.result_object.dummy_results = inner_fold

                # performaceConstraints: DummyEstimator
                if self.constraint_objects is not None:
                    dummy_constraint_objs = [opt for opt in self.constraint_objects if isinstance(opt, DummyPerformance)]
                    if dummy_constraint_objs:
                        for dummy_constraint_obj in dummy_constraint_objs:
                            dummy_constraint_obj.set_dummy_performance(self.result_object.dummy_results)

                return inner_fold
            except Exception as e:
                logger.error(e)
                logger.info("Skipping dummy because of error..")
                return None
        else:
            logger.info("Skipping dummy ..")

