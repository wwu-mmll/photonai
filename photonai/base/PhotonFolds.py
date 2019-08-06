from ..photonlogger.Logger import Logger
from ..base.Helper import PHOTONPrintHelper
from ..base.PhotonPipeline import CacheManager
from ..validation.cross_validation import StratifiedKFoldRegression
from ..validation.Validate import TestPipeline
from ..validation.ResultsDatabase import MDBHelper, FoldOperations, MDBInnerFold, MDBConfig, MDBScoreInformation
from ..optimization.SpeedHacks import DummyPerformance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit

import datetime
import numpy as np
import uuid


class FoldInfo:

    def __init__(self, fold_id=None, fold_nr: int = 0,
                 train_indices: list = None, test_indices: list = None):
        self.fold_id = fold_id
        self.fold_nr = fold_nr
        self.train_indices = train_indices
        self.test_indices = test_indices

    @staticmethod
    def _data_overview(y):
        if len(y.shape) > 1:
            # one hot encoded
            Logger().info("One Hot Encoded data fold information not yet implemented")
            return {}
        else:
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique, counts))

    @staticmethod
    def generate_folds(cv_strategy, X, y=None, groups=None,
                       eval_final_performance=True, test_size=0.2):
        """
        Generates the training and  test set indices for the hyperparameter search
        Returns a tuple of training and test indices

        - If there is a strategy given for the outer cross validation the strategy is called to split the data
            - additionally, if a group variable and a GroupCV is passed, split data according to groups
            - if a group variable and a StratifiedCV is passed, split data according to groups and ignore targets when
            stratifying the data
            - if no group variable but a StratifiedCV is passed, split data according to targets
        - If no strategy is given and eval_final_performance is True, all data is used for training
        - If no strategy is given and eval_final_performance is False: a test set is seperated from the
          training and validation set by the parameter test_size with ShuffleSplit
        """
        # if there is a CV Object for cross validating the hyperparameter search
        if cv_strategy is not None:
            if groups is not None and (isinstance(cv_strategy, (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut))):
                try:
                    data_test_cases = cv_strategy.split(X, y, groups)
                except:
                    Logger().error("Could not split data according to groups")
            elif groups is not None and (isinstance(cv_strategy, (StratifiedKFoldRegression,
                                                                  StratifiedKFold,
                                                                  StratifiedShuffleSplit))):
                try:
                    data_test_cases = cv_strategy.split(X, groups)
                except:
                    Logger().error("Could not stratify data for outer cross validation according to "
                                   "group variable")
            else:
                data_test_cases = cv_strategy.split(X, y)

        # in case we do not want to divide between validation and test set
        # Re eval_final_performance:
        #         # set eval_final_performance to False because
        #         # 1. if no cv-object is given, no split is performed --> seems more logical
        #         #    than passing nothing, passing no cv-object but getting
        #         #    an 80/20 split by default
        #         # 2. if cv-object is given, split is performed but we don't peek
        #         #    into the test set --> thus we can evaluate more hp configs
        #         #    later without double dipping
        elif not eval_final_performance:
            data_test_cases = FoldInfo._yield_all_data()
        # the default is dividing one time into a validation and test set
        else:
            train_test_cv_object = ShuffleSplit(n_splits=1, test_size=test_size)
            data_test_cases = train_test_cv_object.split(X, y)

        fold_objects = list()
        for i, (train_indices, test_indices) in enumerate(data_test_cases):
            fold_info_obj = FoldInfo(fold_id=uuid.uuid4(),
                                     fold_nr=i,
                                     train_indices=train_indices,
                                     test_indices=test_indices)
            fold_objects.append(fold_info_obj)

        return fold_objects

    @staticmethod
    def _yield_all_data(X):
        """
        Helper function that iteratively returns the data stored in self.X
        Returns an iterable version of self.X
        """
        if hasattr(X, 'shape'):
            yield list(range(X.shape[0])), []
        else:
            yield list(range(len(X))), []


class OuterFoldManager:

    def __init__(self, copy_pipe_fnc,
                 optimization_info,
                 outer_fold_id,
                 cross_validation_info,
                 save_predictions: bool=False,
                 save_feature_importances: bool=False,
                 cache_folder=None,
                 cache_updater=None):
        # Information from the Hyperpipe about the design choices
        self.outer_fold_id = outer_fold_id
        self.cross_validaton_info = cross_validation_info
        self.optimization_info = optimization_info
        self.copy_pipe_fnc = copy_pipe_fnc

        self.save_predictions = save_predictions
        self.save_feature_importances = save_feature_importances
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
                                                   self._validation_group,
                                                   **self._validation_kwargs)

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

            hp = TestPipeline(pipe_ctor, current_config,
                              self.optimization_info,
                              self.cross_validaton_info, self.outer_fold_id, self.constraint_objects,
                              save_predictions=self.save_predictions,
                              save_feature_importances=self.save_feature_importances,
                              cache_folder=self.cache_folder,
                              cache_updater=self.cache_updater)

            # Todo: each and everytime the pipe is instantiated just for printing.. that's bad!! WORKAROUND PIPE!!
            example_pipe = pipe_ctor()

            Logger().debug(PHOTONPrintHelper._optimize_printing(example_pipe, current_config))
            Logger().debug('calculating...')

            # Test the configuration cross validated by inner_cv object
            current_config_mdb = hp.fit(self._validation_X, self._validation_y, **self._validation_kwargs)

            current_config_mdb.config_nr = tested_config_counter
            current_config_mdb.config_dict = current_config
            current_config_mdb.human_readable_config = PHOTONPrintHelper.config_to_human_readable_dict(example_pipe, current_config)

            Logger().verbose(PHOTONPrintHelper._optimize_printing(example_pipe, current_config))

            if not current_config_mdb.config_failed:
                # get optimizer_metric and forward to optimizer
                # todo: also pass greater_is_better=True/False to optimizer
                metric_train = MDBHelper.get_metric(current_config_mdb, FoldOperations.MEAN,
                                                    self.optimization_info.best_config_metric)
                metric_test = MDBHelper.get_metric(current_config_mdb, FoldOperations.MEAN,
                                                   self.optimization_info.best_config_metric, train=False)

                if not metric_train or not metric_test:
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
            best_config_outer_fold_mdb = MDBConfig()
            best_config_outer_fold_mdb.config_dict = best_config_outer_fold.config_dict
            best_config_outer_fold_mdb.human_readable_config = best_config_outer_fold.human_readable_config

            # inform user
            Logger().info('Finished hyperparameter optimization!')
            Logger().verbose('Number of tested configurations:' + str(tested_config_counter))
            Logger().verbose('Optimizer metric: ' + self.optimization_info.best_config_metric + '\n' +
                             '   --> Greater is better: ' + str(self.optimization_info.maximize_metric))
            Logger().info('Best config: ' + PHOTONPrintHelper._optimize_printing(example_pipe, best_config_outer_fold_mdb.config_dict))

            # ... and create optimal pipeline
            optimum_pipe = self.copy_pipe_fnc()
            self.cache_updater(optimum_pipe, self.cache_folder, None)
            optimum_pipe.caching = False
            # set self to best config
            optimum_pipe.set_params(**best_config_outer_fold_mdb.config_dict)

            # Todo: set all children to best config and inform to NOT optimize again, ONLY fit
            # for child_name, child_config in best_config_outer_fold_mdb.children_config_dict.items():
            #     if child_config:
            #         # in case we have a pipeline stacking we need to identify the particular subhyperpipe
            #         splitted_name = child_name.split('__')
            #         if len(splitted_name) > 1:
            #             stacking_element = self.optimum_pipe.named_steps[splitted_name[0]]
            #             pipe_element = stacking_element.pipe_elements[splitted_name[1]]
            #         else:
            #             pipe_element = self.optimum_pipe.named_steps[child_name]
            #         pipe_element.set_params(**child_config)
            #         pipe_element.is_final_fit = True

            # self.__distribute_cv_info_to_hyperpipe_children(reset=True)

            Logger().verbose('...now fitting with optimum configuration')
            fit_time_start = datetime.datetime.now()
            optimum_pipe.fit(self._validation_X, self._validation_y, **self._validation_kwargs)

            best_config_outer_fold_mdb.fit_duration_minutes = (datetime.datetime.now() - fit_time_start).total_seconds() / 60
            self.result_object.best_config = best_config_outer_fold_mdb
            self.result_object.best_config.inner_folds = []

            # save test performance
            best_config_performance_mdb = MDBInnerFold()
            best_config_performance_mdb.fold_nr = 1
            best_config_performance_mdb.number_samples_training = self._validation_y.shape[0]
            best_config_performance_mdb.number_samples_validation = self._test_y.shape[0]

            if self.cross_validaton_info.eval_final_performance:
                # Todo: generate mean and std over outer folds as well. move this items to the top
                Logger().verbose('...now predicting unseen data on test set')

                test_score_mdb = TestPipeline.score(optimum_pipe, self._test_X, self._test_y,
                                                    indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices,
                                                    metrics=self.optimization_info.metrics,
                                                    save_predictions=self.save_predictions,
                                                    save_feature_importances=self.save_feature_importances,
                                                    **self._test_kwargs)

                Logger().info('.. calculating metrics for test set')
                Logger().verbose('...now predicting final model with training data')

                train_score_mdb = TestPipeline.score(optimum_pipe, self._validation_X, self._validation_y,
                                                     indices=self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices,
                                                     metrics=self.optimization_info.metrics,
                                                     save_predictions=self.save_predictions,
                                                     save_feature_importances=self.save_feature_importances,
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

            self.result_object.best_config.inner_folds = [best_config_performance_mdb]

        Logger().info('This took {} minutes.'.format((datetime.datetime.now() - outer_fold_fit_start_time).total_seconds() / 60))

        if self.cache_folder is not None:
            Logger().info("Clearing Cache")
            CacheManager.clear_cache_files(self.cache_folder)

    def fit_dummy(self, X, y, dummy):

        try:
            train_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]
            train_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].train_indices]

            if isinstance(train_X, np.ndarray):
                if len(train_X.shape) > 2:
                    Logger().info("Skipping dummy estimator because of too much dimensions")

            dummy.fit(train_X, train_y)
            train_scores = TestPipeline.score(dummy, train_X, train_y, metrics=self.optimization_info.metrics)

            # fill result tree with fold information
            inner_fold = MDBInnerFold()
            inner_fold.training = train_scores

            if self.cross_validaton_info.eval_final_performance:
                test_X = X[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]
                test_y = y[self.cross_validaton_info.outer_folds[self.outer_fold_id].test_indices]
                test_scores = TestPipeline.score(dummy, test_X, test_y, metrics=self.optimization_info.metrics)
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
