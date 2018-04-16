import time
from hashlib import sha1
from itertools import product
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.pipeline import Pipeline
from pymodm import connect

from Framework.Register import PhotonRegister
from Logging.Logger import Logger
from .OptimizationStrategies import GridSearchOptimizer, RandomGridSearchOptimizer, TimeBoxedRandomGridSearchOptimizer
from .ResultLogging import *
from .Validation import TestPipeline, OptimizerMetric


class Hyperpipe(BaseEstimator):
    """
    Wrapper class for machine learning pipeline, holding all pipeline elements
    and managing the optimization of the hyperparameters
    """


    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                            'random_grid_search': RandomGridSearchOptimizer,
                            'timeboxed_random_grid_search': TimeBoxedRandomGridSearchOptimizer}

    def __init__(self, name, inner_cv: BaseCrossValidator,
                 optimizer='grid_search', optimizer_params: dict = None, local_search: bool =True,
                 groups=None, config=None, overwrite_x=None, overwrite_y=None,
                 metrics=None, best_config_metric=None, outer_cv=None,
                 test_size=0.2, eval_final_performance=False, save_all_predictions: bool=False, debug_cv_mode=False,
                 set_random_seed: bool=False,  filter_element=None,
                 logfile: str = '', logging: bool =False, verbose:int =0):

        """
        Setup the hyperparameter search, the nested cross validation and other parameters of the pipeline construct

        Keyword arguments: 
            :param name: str -- name of hyperpipe construct 
            :param inner_cv: BaseCrossValidator -- cross validation strategy to test hyperparameter configurations, generates the validation set
            :param outer_cv: BaseCrossValidator -- cross validation strategy to use for the hyperparameter search itself, generates the test set
            :param optimizer: str or object -- hyperparameter optimization algorithm (default 'grid_search')
            :param optimizer_params: dict -- parameters to pass to the optimizer
            :param metrics: list of str -- metrics that should be calculated for both validation and test set
            :param best_config_metric: str -- the metric that should be maximized or minimized in order to choose the best hyperparameter configuration
            :param eval_final_performance: bool -- if the metrics should be calculated for the test set (default False)
            :param test_size: double -- the amount of the data that should be left out if no outer_cv is given and eval_final_perfomance is set to True (default 0.2)
            :param set_random_seed: bool -- if true sets the random seed to 42 (default False)
            :param verbose: int -- the level of verbosity (default 0)
            :param logging: bool -- if the output should be printed to a log file (default False)
            :param groups: list -- info for advanced cross validation strategies about each row in data
            :param local_search: bool -- if the hyperpipe should optimize the hyperparameters of its element (default True)
            :param filter_element: SourceSpltiter - transforms the input data, e.g. extracts certain columns
            :param imbalanced_data_strategy_filter: OverOrUnderSampler - balances the data if there are imbalanced classes
            :param overwrite_x: list -- data that is used when the hyperpipe object is an element of another hyperpipe
            :param overwrite_y: list -- targets that are used when the hyperpipe object is an element of another hyperpipe
            :param config: construct pipeline elements and their hyperparameters and default parameters from JSON
            :param debug_cv_mode: bool -- boolean for the internal unit tests of the nested cross validation
            :param save_all_predictions: bool -- specify whether to save test/validation/training predictions for all hyperparameter configurations

        Example:
            manager = Hyperpipe('test_manager',
                                optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1},
                                outer_cv=ShuffleSplit(test_size=0.2, n_splits=1),
                                inner_cv=KFold(n_splits=10, shuffle=True),
                                metrics=['accuracy', 'precision', 'recall', "f1_score"],
                                best_config_metric='accuracy', eval_final_performance=True, 
                                logging=True, verbose=2)


        """

        # Re eval_final_performance:
        # set eval_final_performance to False because
        # 1. if no cv-object is given, no split is performed --> seems more logical
        #    than passing nothing, passing no cv-object but getting
        #    an 80/20 split by default
        # 2. if cv-object is given, split is performed but we don't peek
        #    into the test set --> thus we can evaluate more hp configs
        #    later without double dipping

        if optimizer_params is None:
            optimizer_params = {}
        self.fit_duration = 0
        self.fold_list = []
        self.name = name
        self.hyperparameter_specific_config_cv_object = inner_cv
        self.cv_iter = None
        self.X = None
        self.y = None
        self.groups = groups
        self.filter_element = filter_element

        self.data_test_cases = None
        self.config_history = []
        self.performance_history = []
        self.children_config_setup = []
        self.best_config = None
        self.best_children_config = None
        self.best_performance = None
        self.is_final_fit = False
        self.save_all_prediction = save_all_predictions

        self.debug_cv_mode = debug_cv_mode
        self.logging = logging
        if set_random_seed:
            import random
            random.seed(42)
            print('set random seed to 42')
        self.verbose = verbose
        Logger().set_verbosity(self.verbose)
        if logfile:
            Logger().set_custom_log_file(logfile)

        self.pipeline_elements = []
        self.pipeline_param_list = {}
        self.pipe = None
        self.optimum_pipe = None
        self.metrics = metrics
        self.best_config_metric = best_config_metric
        self.config_optimizer = None

        self.result_tree = None
        self.mother_outer_fold_counter = 0
        self.mother_inner_fold_counter = 0
        self.mother_config_counter = 0

        # Todo: this might be a case for sanity checking
        self.overwrite_x = overwrite_x
        self.overwrite_y = overwrite_y

        self._hyperparameters = []
        self._config_grid = []

        # containers for optimization history and Logging
        self.config_history = []
        self.performance_history_list = []
        self.parameter_history = []
        self.test_performances = {}

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

        if isinstance(optimizer, str):
            # instantiate optimizer from string
            #  Todo: check if optimizer strategy is already implemented
            optimizer_class = self.OPTIMIZER_DICTIONARY[optimizer]
            optimizer_instance = optimizer_class(**optimizer_params)
            self.optimizer = optimizer_instance
            # we need an object for global search
            # so with a string it must be local search
            self.local_search = True
        else:
            # Todo: check if correct object
            self.optimizer = optimizer
            self.local_search = local_search

        self.test_size = test_size
        self.validation_X = None
        self.validation_y = None
        self.test_X = None
        self.test_y = None
        self.eval_final_performance = eval_final_performance
        self.hyperparameter_fitting_cv_object = outer_cv
        self.last_fit_data_hash = None
        self.current_fold = -1
        self.num_of_folds = 0
        self.is_mother_pipe = True
        self.fold_data_hashes = []

    def __iadd__(self, pipe_element):
        """
        Add an element to the machine learning pipeline

        :param pipe_element: PipelineElement
        :return: self
        """
        # if isinstance(pipe_element, PipelineElement):
        self.pipeline_elements.append(pipe_element)
        # Todo: is repeated each time element is added....
        self.prepare_pipeline()
        return self
        # else:
        #     Todo: raise error
        # raise TypeError("Element must be of type Pipeline Element")


    @property
    def hyperparameters(self):
        return self._hyperparameters


    def add(self, pipe_element):
        self.__iadd__(pipe_element)


    def yield_all_data(self):
        """
        helper that iteratively returns the data stored in self.X
        :return: self.X
        """
        if hasattr(self.X, 'shape'):
            yield list(range(self.X.shape[0])), []
        else:
            yield list(range(len(self.X))), []


    def generate_outer_cv_indices(self):
        """
        generates the training and  test set indices for the hyperparameter search

        if there is a strategy given for the outer cross validation it is used
        if no strategy is given and eval_final_performance is True, all data is used for training
        else: a test set is generated by the parameter test_size with ShuffleSplit
        :return: list of (training and, test indices)
        """
        # if there is a CV Object for cross validating the hyperparameter search
        if self.hyperparameter_fitting_cv_object:
            self.data_test_cases = self.hyperparameter_fitting_cv_object.split(self.X, self.y)
        # in case we do not want to divide between validation and test set
        elif not self.eval_final_performance:
            self.data_test_cases = self.yield_all_data()
        # the default is dividing one time into a validation and test set
        else:
            train_test_cv_object = ShuffleSplit(n_splits=1, test_size=self.test_size)
            self.data_test_cases = train_test_cv_object.split(self.X, self.y)


    def distribute_cv_info_to_hyperpipe_children(self, num_of_folds=None, reset=False, reset_final_fit=False,
                                                 outer_fold_counter=None, inner_fold_counter=None, config_counter=None):

        """
        informs all elements of the pipeline that are of type hyperpipe (hyperpipe children)
        about the mother's configuration or current state

        :param num_of_folds: how many inner_folds the superior hyperpipe has
        :param reset: if the hyperparameter search starts anew
        """

        def _distribute_info_to_object(pipe_object, number_of_folds, reset_folds, reset_final_fit,
                                      outer_fold_counter, inner_fold_counter, config_counter):
            if pipe_object.local_search:
                if number_of_folds is not None:
                    pipe_object.num_of_folds = number_of_folds
                    pipe_object.is_mother_pipe = False
                if reset_folds:
                    pipe_object.current_fold = -1
                if outer_fold_counter is not None:
                    pipe_object.mother_outer_fold_counter = outer_fold_counter
                if inner_fold_counter is not None:
                    pipe_object.mother_inner_fold_counter = inner_fold_counter
                if config_counter:
                    pipe_object.mother_config_counter = config_counter
                if reset_final_fit:
                    pipe_object.is_final_fit = False

        for element_tuple in self.pipe.steps:
            element_object = element_tuple[1]
            if isinstance(element_object, Hyperpipe):
                _distribute_info_to_object(element_object, num_of_folds, reset, reset_final_fit,
                                          outer_fold_counter, inner_fold_counter, config_counter)
            elif isinstance(element_object, PipelineStacking):
                for child_pipe_name, child_pipe_object in element_object.pipe_elements.items():
                    _distribute_info_to_object(child_pipe_object, num_of_folds, reset, reset_final_fit,
                                              outer_fold_counter, inner_fold_counter, config_counter)

    def update_mother_inner_fold_nr(self, new_inner_fold_nr):
        self.distribute_cv_info_to_hyperpipe_children(inner_fold_counter=new_inner_fold_nr)

    def fit(self, data, targets, **fit_params):
        """
        Starts the hyperparameter search and/or fits the pipeline to the data and targets

        Manages the nested cross validated hyperparameter search:
        1. requests new configurations from the hyperparameter search strategy, the optimizer,
        2. initializes the testing of a specific configuration,
        3. communicates the result to the optimizer,
        4. repeats 1-3 until optimizer delivers no more configurations to test
        5. finally searches for the best config,
        6. trains the pipeline with the best config and evaluates the performance on the test set

        :param data: the training and test data
        :param targets: the truth value
        :param fit_params: any param dict
        :return: self
        """

        # in case we want to inject some data from outside the pipeline
        if self.overwrite_x is None and self.overwrite_y is None:
            self.X = data
            self.y = targets
        else:
            self.X = self.overwrite_x
            self.y = self.overwrite_y

        # !!!!!!!!!!!!!!!! FIT ONLY IF DATA CHANGED !!!!!!!!!!!!!!!!!!!
        # -------------------------------------------------------------

        # in case we need to reduce the dimension of the data due to parallelity of the outer pipe, lets do it.
        if self.filter_element:
            self.X = self.filter_element.transform(self.X)

        self.current_fold += 1

        # be compatible to list of (image-) files
        if isinstance(self.X, list):
            self.X = np.asarray(self.X)
        if isinstance(self.y, list):
            self.y = np.asarray(self.y)
        #if not isinstance(self.X, np.ndarray): # and isinstance(self.X[0], str):
        #    self.X = np.asarray(self.X)

        # handle PhotonNeuro Imge paths as data
        # ToDo: Need to check the DATA, not the img paths for PhotonNeuro
        new_data_hash = sha1(np.asarray(self.X, order='C')).hexdigest()

        # fit
        # 1. if it is first time ever or
        # 2. the data did change for that fold or
        # 3. if it is the mother pipe (then number_of_folds = 0)
        if (len(self.fold_data_hashes) < self.num_of_folds) \
                or (self.num_of_folds > 0 and self.fold_data_hashes[self.current_fold] != new_data_hash) \
                or self.num_of_folds == 0:

            # save data hash for that fold
            if self.num_of_folds > 0:
                if len(self.fold_data_hashes) < self.num_of_folds:
                    self.fold_data_hashes.append(new_data_hash)
                else:
                    self.fold_data_hashes[self.current_fold] = new_data_hash

            # optimize: iterate through configs and save results
            if self.local_search and not self.is_final_fit:

                # first check if correct optimizer metric has been chosen
                # pass pipeline_elements so that OptimizerMetric can look for last
                # element and use the corresponding score method
                self.config_optimizer = OptimizerMetric(self.best_config_metric, self.pipeline_elements, self.metrics)
                self.metrics = self.config_optimizer.check_metrics()

                if 'score' in self.metrics:
                    Logger().warn('Attention: Scoring with default score function of estimator can slow down calculations!')

                # generate OUTER ! cross validation splits to iterate over
                self.generate_outer_cv_indices()

                outer_fold_counter = 0

                if not self.is_mother_pipe:
                    self.result_tree_name = self.name + '_outer_fold_' + str(self.mother_outer_fold_counter)  \
                                            + '_inner_fold_' + str(self.mother_inner_fold_counter)
                else:
                    self.result_tree_name = self.name

                # initialize result logging with hyperpipe class
                self.result_tree = MDBHyperpipe(name=self.result_tree_name)
                self.result_tree.outer_folds = []

                # loop over outer cross validation
                for train_indices, test_indices in self.data_test_cases:

                    # give the optimizer the chance to inform about elements
                    self.optimizer.prepare(self.pipeline_elements)
                    self.performance_history_list = []

                    outer_fold_counter += 1
                    outer_fold_fit_start_time = time.time()

                    Logger().info('HYPERPARAMETER SEARCH OF {0}, Outer Cross Validation Fold {1}'
                                  .format(self.name, outer_fold_counter))

                    t1 = time.time()

                    # Prepare Train and Validation set data
                    self.validation_X = self.X[train_indices]
                    self.validation_y = self.y[train_indices]
                    self.test_X = self.X[test_indices]
                    self.test_y = self.y[test_indices]

                    # Prepare inner cross validation
                    cv_iter = list(self.hyperparameter_specific_config_cv_object.split(self.validation_X, self.validation_y))
                    num_folds = len(cv_iter)
                    num_samples_train = len(self.validation_y)
                    num_samples_test = len(self.test_y)

                    # distribute number of folds to encapsulated child hyperpipes
                    self.distribute_cv_info_to_hyperpipe_children(num_of_folds=num_folds,
                                                                  outer_fold_counter=outer_fold_counter)

                    tested_config_counter = 0

                    # add outer fold info object to result tree
                    outer_fold = MDBOuterFold(fold_nr=outer_fold_counter)
                    outer_fold.tested_config_list = []
                    self.result_tree.outer_folds.append(outer_fold)

                    # do the optimizing
                    for specific_config in self.optimizer.next_config:

                        # try:
                        #     loaded_result_tree = list(MDBHyperpipe.objects.raw({'_id':self.result_tree_name}))[0]
                        #     config_item = loaded_result_tree.outer_folds[outer_fold_counter-1].tested_config_list[tested_config_counter]
                        #
                        #     tested_config_counter += 1
                        #     self.distribute_cv_info_to_hyperpipe_children(reset=True,
                        #                                                   config_counter=tested_config_counter)
                        #
                        #     config_score = (
                        #     MDBHelper.get_metric(config_item, FoldOperations.MEAN, self.config_optimizer.metric),
                        #     MDBHelper.get_metric(config_item, FoldOperations.MEAN,
                        #                          self.config_optimizer.metric, train=False))
                        #     self.performance_history_list.append(config_score)
                        #
                        #     # save the configuration of all children pipelines
                        #     children_config = {}
                        #     children_config_ref_list = []
                        #     for pipe_step in self.pipe.steps:
                        #         item = pipe_step[1]
                        #         if isinstance(item, Hyperpipe):
                        #             if item.local_search and item.best_config is not None:
                        #                 children_config[item.name] = item.best_config
                        #         elif isinstance(item, PipelineStacking):
                        #             for subhyperpipe_name, hyperpipe in item.pipe_elements.items():
                        #                 if hyperpipe.local_search and hyperpipe.best_config is not None:
                        #                     # special case: we need to access pipe over pipeline_stacking element
                        #                     children_config[item.name + '__' + subhyperpipe_name] = hyperpipe.best_config.config_dict
                        #                     # children_config_ref_list.append(hyperpipe.best_config._id)
                        #     specific_parameters = self.pipe.get_params()
                        #     #config_item.full_model_spec = specific_parameters
                        #
                        #     config_item.children_config_dict = children_config
                        #     config_item.children_config_ref = children_config_ref_list
                        #     self.result_tree.outer_folds[outer_fold_counter-1].tested_config_list.append(config_item)
                        #     Logger().debug('optimizing of:' + self.name)
                        #     Logger().debug(self.optimize_printing(specific_config))
                        #     Logger().info('Loading results for this config from MongoDB')
                        # except:
                        self.distribute_cv_info_to_hyperpipe_children(reset=True, config_counter=tested_config_counter)
                        hp = TestPipeline(self.pipe, specific_config, self.metrics, self.update_mother_inner_fold_nr)
                        Logger().debug('optimizing of:' + self.name)
                        Logger().debug(self.optimize_printing(specific_config))
                        Logger().debug('calculating...')

                        # Test the configuration cross validated by inner_cv object
                        config_item = hp.calculate_cv_score(self.validation_X, self.validation_y, cv_iter,
                                          save_predictions=self.save_all_predictions)
                        config_item.config_nr = tested_config_counter
                        config_item.config_dict = specific_config
                        config_item.pipe_name = self.name
                        tested_config_counter += 1
                        config_item.human_readable_config = self.config_to_dict(specific_config)

                        # save the configuration of all children pipelines
                        children_config = {}
                        children_config_ref_list = []
                        for pipe_step in self.pipe.steps:
                            item = pipe_step[1]
                            if isinstance(item, Hyperpipe):
                                if item.local_search and item.best_config is not None:
                                    children_config[item.name] = item.best_config
                            elif isinstance(item, PipelineStacking):
                                for subhyperpipe_name, hyperpipe in item.pipe_elements.items():
                                    if hyperpipe.local_search and hyperpipe.best_config is not None:
                                        # special case: we need to access pipe over pipeline_stacking element
                                        children_config[item.name + '__' + subhyperpipe_name] = hyperpipe.best_config.config_dict
                                        # children_config_ref_list.append(hyperpipe.best_config._id)
                        specific_parameters = self.pipe.get_params()
                        #config_item.full_model_spec = specific_parameters

                        config_item.children_config_dict = children_config
                        config_item.children_config_ref = children_config_ref_list

                        Logger().verbose(self.optimize_printing(specific_config))

                        if not config_item.config_failed:
                            # get optimizer_metric and forward to optimizer
                            # todo: also pass greater_is_better=True/False to optimizer
                            config_score = (MDBHelper.get_metric(config_item, FoldOperations.MEAN, self.config_optimizer.metric),
                                            MDBHelper.get_metric(config_item, FoldOperations.MEAN,
                                                                 self.config_optimizer.metric, train=False))

                            # Print Result for config
                            Logger().debug('...done:')
                            Logger().verbose(self.config_optimizer.metric + str(config_score))
                        else:
                            config_score = (-1, -1)
                            # Print Result for config
                            Logger().debug('...failed:')
                            Logger().error(config_item.config_error)

                        self.performance_history_list.append(config_score)

                        # add config to result tree and do intermediate saving
                        self.result_tree.outer_folds[-1].tested_config_list.append(config_item)
                        self.result_tree.save()

                        # 3. inform optimizer about performance
                        self.optimizer.evaluate_recent_performance(specific_config, config_score)

                    # Todo: Do better error checking
                    if len(self.performance_history_list) > 0:
                        best_train_config = self.config_optimizer.get_optimum_config(outer_fold.tested_config_list)

                        best_config_item_test = MDBConfig()
                        best_config_item_test.children_config_dict = best_train_config.children_config_dict
                        best_config_item_test.pipe_name = self.name
                        best_config_item_test.children_config_ref = best_train_config.children_config_ref
                        # best_config_item_test.best_config_ref_to_train_item = best_train_config._id
                        best_config_item_test.config_dict = best_train_config.config_dict
                        best_config_item_test.human_readable_config = best_train_config.human_readable_config
                        self.best_config = best_config_item_test

                        # inform user
                        Logger().info('finished optimization of ' + self.name)
                        Logger().verbose('Result')
                        Logger().verbose('Number of tested configurations:' +
                                         str(len(self.performance_history_list)))
                        Logger().verbose('Optimizer metric: ' + self.config_optimizer.metric + '\n' +
                                         '   --> Greater is better: ' + str(self.config_optimizer.greater_is_better))
                        Logger().info('Best config: ' + self.optimize_printing(self.best_config.config_dict) +
                                      '\n' + '... with children config: '
                                      + self.optimize_printing(self.best_config.children_config_dict))

                        # ... and create optimal pipeline
                        self.optimum_pipe = self.pipe
                        # set self to best config
                        self.optimum_pipe.set_params(**self.best_config.config_dict)

                        # set all children to best config and inform to NOT optimize again, ONLY fit
                        for child_name, child_config in self.best_config.children_config_dict.items():
                            if child_config:
                                # in case we have a pipeline stacking we need to identify the particular subhyperpipe
                                splitted_name = child_name.split('__')
                                if len(splitted_name) > 1:
                                    stacking_element = self.optimum_pipe.named_steps[splitted_name[0]]
                                    pipe_element = stacking_element.pipe_elements[splitted_name[1]]
                                else:
                                    pipe_element = self.optimum_pipe.named_steps[child_name]
                                pipe_element.set_params(**child_config)
                                pipe_element.is_final_fit = True

                        self.distribute_cv_info_to_hyperpipe_children(reset=True)

                        Logger().verbose('...now fitting ' + self.name + ' with optimum configuration')
                        fit_time_start = time.time()
                        self.optimum_pipe.fit(self.validation_X, self.validation_y)
                        final_fit_duration = time.time() - fit_time_start

                        #self.best_config.full_model_spec = self.optimum_pipe.get_params()
                        self.best_config.fit_duration_minutes = final_fit_duration
                        self.result_tree.outer_folds[-1].best_config = self.best_config

                        if not self.debug_cv_mode and self.eval_final_performance:
                            # Todo: generate mean and std over outer folds as well. move this items to the top
                            Logger().verbose('...now predicting ' + self.name + ' unseen data')

                            final_fit_test_item = TestPipeline.score(self.optimum_pipe, self.test_X, self.test_y,
                                                                     self.metrics, save_predictions=True)

                            Logger().info('.. calculating metrics for test set (' + self.name + ')')
                            Logger().verbose('...now predicting ' + self.name + ' final model with training data')

                            final_fit_train_item = TestPipeline.score(self.optimum_pipe, self.validation_X, self.validation_y,
                                                                      self.metrics, save_predictions=True)

                            # save test fold
                            test_set_fold = MDBInnerFold()
                            test_set_fold.fold_nr = 1
                            test_set_fold.number_samples_training = num_samples_train
                            test_set_fold.number_samples_validation = num_samples_test
                            test_set_fold.training = final_fit_train_item
                            test_set_fold.validation = final_fit_test_item
                            self.result_tree.outer_folds[-1].best_config.inner_folds.append(test_set_fold)


                            Logger().info('PERFORMANCE TRAIN:')
                            for m_key, m_value in final_fit_train_item.metrics.items():
                                Logger().info(str(m_key) + ": " + str(m_value))

                            Logger().info('PERFORMANCE TEST:')
                            for m_key, m_value in final_fit_test_item.metrics.items():
                                    Logger().info(str(m_key) + ": " + str(m_value))


                    Logger().info('This took {} minutes.'.format((time.time() - t1) / 60))
                    self.result_tree.save()
                    self.distribute_cv_info_to_hyperpipe_children(reset_final_fit=True, outer_fold_counter=outer_fold_counter)

                # save result tree to db
                Logger().info("Saved result tree to database")
                self.result_tree.save()
                if self.logging:
                    self.result_tree.print_csv_file(self.name + "_" + str(time.time()) + ".csv")
            ###############################################################################################
            else:
                self.pipe.fit(self.X, self.y, **fit_params)

        else:
            Logger().verbose("Avoided fitting of " + self.name + " on fold "
                             + str(self.current_fold) + " because data did not change")
            Logger().verbose('Best config of ' + self.name + ' : ' + str(self.best_config))

        return self

    def predict(self, data):
        # Todo: if local_search = true then use optimized pipe here?
        if self.pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.predict(data)

    def predict_proba(self, data):
        """
        Predict probabilities
        :param data: array-like
        :type data: float
        :return: predicted values, array
        """
        if self.pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.predict_proba(data)

    def transform(self, data):
        if self.pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.transform(data)

    def get_params(self, deep=True):
        if self.pipe is not None:
            return self.pipe.get_params(deep)
        else:
            return None

    def set_params(self, **params):
        if self.pipe is not None:
            self.pipe.set_params(**params)
        return self

    def prepare_pipeline(self):
        # prepare pipeline, hyperparams and config-grid
        self._config_grid = []
        self._hyperparameters = []
        pipeline_steps = []
        all_hyperparams = {}
        all_config_grids = []
        for item in self.pipeline_elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))
            all_hyperparams[item.name] = item.hyperparameters
            if item.config_grid:
                all_config_grids.append(item.config_grid)
        self._hyperparameters = all_hyperparams
        if len(all_config_grids) == 1:
            self._config_grid = all_config_grids[0]
        elif all_config_grids:
            # unpack list of dictionaries in one dictionary
            tmp_config_grid = list(product(*all_config_grids))
            for config_iterable in tmp_config_grid:
                base = dict(config_iterable[0])
                for i in range(1, len(config_iterable)):
                    base.update(config_iterable[i])
                self._config_grid.append(base)

        # build pipeline...
        self.pipe = Pipeline(pipeline_steps)

    def copy_me(self):
        item_list =[]
        for item in self.pipeline_elements:
            item_list.append(item.copy_me())
        return item_list

    def _copy_pipeline(self):
        pipeline_steps = []
        for item in self.pipeline_elements:
            cpy = item.copy_me()
            if isinstance(cpy, list):
                for new_step in cpy:
                    pipeline_steps.append((new_step.name, new_step))
            else:
                pipeline_steps.append((cpy.name, cpy))
        return Pipeline(pipeline_steps)

    def inverse_transform_pipeline(self, hyperparameters, data, targets, data_to_inverse):
        copied_pipe = self._copy_pipeline()
        copied_pipe.set_params(**hyperparameters)
        copied_pipe.fit(data, targets)
        return copied_pipe.inverse_transform(data_to_inverse)

    def optimize_printing(self, config):
        prettified_config = [self.name + '\n']
        for el_key, el_value in config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in self.pipe.named_steps:
                new_pretty_key = '    ' + name + '->'
                prettified_config.append(new_pretty_key +
                                         self.pipe.named_steps[name].prettify_config_output(rest, el_value) + '\n')
            else:
                Logger().error('ValueError: Item is not contained in pipeline:' + name)
                raise ValueError('Item is not contained in pipeline:' + name)
        return ''.join(prettified_config)

    @staticmethod
    def prettify_config_output(config_name, config_value):
        if config_name == "disabled" and config_value is False:
            return "enabled = True"
        else:
            return config_name + '=' + str(config_value)

    @property
    def config_grid(self):
        return self._config_grid

    def create_pipeline_elements_from_config(self, config):
        # Todo: Not reassign 'self'!!!
        for key, all_params in config.items():
            self += PipelineElement(key, all_params, {})

    def config_to_dict(self, specific_config):
        config = {}
        for key, value in specific_config.items():
            items = key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in self.pipe.named_steps:
                config.update(self.pipe.named_steps[name].prettify_config_output(rest, value, return_dict=True))
                #config[name] = value
        return config


class SourceFilter(BaseEstimator):
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, self.indices]


class PipelineElement(BaseEstimator):
    # Registering Pipeline Elements
    ELEMENT_DICTIONARY = PhotonRegister.get_package_info(['PhotonCore', 'PhotonNeuro'])

    @classmethod
    def create(cls, name, hyperparameters=None, test_disabled=False, disabled=False, **kwargs):
        if hyperparameters is None:
            hyperparameters = {}
        if name in PipelineElement.ELEMENT_DICTIONARY:
            try:
                desired_class_info = PipelineElement.ELEMENT_DICTIONARY[name]
                desired_class_home = desired_class_info[0]
                desired_class_name = desired_class_info[1]
                imported_module = __import__(desired_class_home, globals(), locals(), desired_class_name, 0)
                desired_class = getattr(imported_module, desired_class_name)
                base_element = desired_class(**kwargs)
                obj = PipelineElement(name, base_element, hyperparameters, test_disabled, disabled)
                return obj
            except AttributeError as ae:
                Logger().error('ValueError: Could not find according class:'
                               + str(PipelineElement.ELEMENT_DICTIONARY[name]))
                raise ValueError('Could not find according class:', PipelineElement.ELEMENT_DICTIONARY[name])
        else:
            Logger().error('Element not supported right now:' + name)
            raise NameError('Element not supported right now:', name)

    def copy_me(self):
        return deepcopy(self)

    def __init__(self, name, base_element, hyperparameters: dict, test_disabled=False, disabled=False):
        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized --> sklearn: get_params.keys
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.base_element = base_element
        self.disabled = disabled
        self.test_disabled = test_disabled
        self._hyperparameters = hyperparameters
        self._sklearn_hyperparams = {}
        self._sklearn_disabled = self.name + '__disabled'
        self._config_grid = []
        self.hyperparameters = self._hyperparameters

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        # Todo: Make sure that set_disabled is not included when generating config_grid and stuff
        self._hyperparameters = value
        self.generate_sklearn_hyperparameters()
        self.generate_config_grid()
        if self.test_disabled:
            self._hyperparameters.update({'test_disabled': True})

    @property
    def config_grid(self):
        return self._config_grid

    @property
    def sklearn_hyperparams(self):
        return self._sklearn_hyperparams

    def generate_sklearn_hyperparameters(self):
        self._sklearn_hyperparams = {}
        for attribute, value_list in self._hyperparameters.items():
            self._sklearn_hyperparams[self.name + '__' + attribute] = value_list

    def generate_config_grid(self):
        for item in ParameterGrid(self.sklearn_hyperparams):
            if self.test_disabled:
                item[self._sklearn_disabled] = False
            self._config_grid.append(item)
        if self.test_disabled:
            self._config_grid.append({self._sklearn_disabled: True})

    def get_params(self, deep=True):
        return self.base_element.get_params(deep)

    def set_params(self, **kwargs):
        # element disable is a construct used for this container only
        if self._sklearn_disabled in kwargs:
            self.disabled = kwargs[self._sklearn_disabled]
            del kwargs[self._sklearn_disabled]
        elif 'disabled' in kwargs:
            self.disabled = kwargs['disabled']
            del kwargs['disabled']
        self.base_element.set_params(**kwargs)
        return self

    def fit(self, data, targets=None):
        if not self.disabled:
            obj = self.base_element
            obj.fit(data, targets)
            # self.base_element.fit(data, targets)
        return self

    def predict(self, data, targets=None):
        if not self.disabled:
            if hasattr(self.base_element, 'predict'):
                return self.base_element.predict(data)
            elif hasattr(self.base_element, 'transform'):
                return self.base_element.transform(data)
            else:
                Logger().error('BaseException. Base Element should have function ' +
                               'predict, or at least transform.')
                raise BaseException('Base Element should have function predict, or at least transform.')
        else:
            return data

    def predict_proba(self, data):
        """
        Predict probabilities
        Base element needs predict_proba() function, otherwise throw
        base exception.
        :param data: array-like
        :type data: float
        :return: predicted values, array
        """
        if not self.disabled:
            if hasattr(self.base_element, 'predict_proba'):
                return self.base_element.predict_proba(data)
            else:
                Logger().error('BaseException. Base Element should have "predict_proba" function.')
            raise BaseException('Base Element should have predict_proba function.')
        return data

    # def fit_predict(self, data, targets):
    #     if not self.disabled:
    #         return self.base_element.fit_predict(data, targets)
    #     else:
    #         return data

    def transform(self, data, targets=None):
        if not self.disabled:
            if hasattr(self.base_element, 'transform'):
                return self.base_element.transform(data)
            elif hasattr(self.base_element, 'predict'):
                return self.base_element.predict(data)
            else:
                Logger().error('BaseException: transform-predict-mess')
                raise BaseException('transform-predict-mess')
        else:
            return data

    def inverse_transform(self, data):
        if hasattr(self.base_element, 'inverse_transform'):
            return self.base_element.inverse_transform(data)
        else:
            # raise Warning('Element ' + self.name + ' has no method inverse_transform')
            return data

    # def fit_transform(self, data, targets=None):
    #     if not self.disabled:
    #         if hasattr(self.base_element, 'fit_transform'):
    #             return self.base_element.fit_transform(data, targets)
    #         elif hasattr(self.base_element, 'transform'):
    #             self.base_element.fit(data, targets)
    #             return self.base_element.transform(data)
    #         # elif hasattr(self.base_element, 'predict'):
    #         #     self.base_element.fit(data, targets)
    #         #     return self.base_element.predict(data)
    #     else:
    #         return data

    def score(self, X_test, y_test):
        return self.base_element.score(X_test, y_test)

    def prettify_config_output(self, config_name, config_value, return_dict=False):
        if config_name == "disabled" and config_value is False:
            if return_dict:
                return {'enabled':True}
            else:
                return "enabled = True"
        else:
            if return_dict:
                return {config_name:config_value}
            else:
                return config_name + '=' + str(config_value)


class PipelineStacking(PipelineElement):
    def __init__(self, name, pipeline_fusion_elements, voting=True):
        super(PipelineStacking, self).__init__(name, None, hyperparameters={}, test_disabled=False, disabled=False)

        self._hyperparameters = {}
        self._config_grid = []
        self.pipe_elements = OrderedDict()
        self.voting = voting

        all_config_grids = []
        for item in pipeline_fusion_elements:
            self.pipe_elements[item.name] = item
            self._hyperparameters[item.name] = item.hyperparameters

            # we want to communicate the configuration options to the optimizer, when local_search = False
            # but not when the item takes care of itself, that is, when local_search = True
            add_item_config_grid = True
            if hasattr(item, 'local_search'):
                if item.local_search:
                    add_item_config_grid = False

            # for each configuration
            if add_item_config_grid:
                tmp_config_grid = []
                for config in item.config_grid:
                    # # for each configuration item:
                    # # if config is no dictionary -> unpack it
                    if config:
                        tmp_dict = dict(config)
                        tmp_config = dict(config)
                        for key, element in tmp_config.items():
                            # update name to be referable to pipeline
                            if isinstance(item, PipelineElement):
                                tmp_dict[self.name + '__' + key] = tmp_dict.pop(key)
                            else:
                                tmp_dict[self.name + '__' + item.name + '__' + key] = tmp_dict.pop(key)
                        tmp_config_grid.append(tmp_dict)
                if tmp_config_grid:
                    all_config_grids.append(tmp_config_grid)
        if all_config_grids:
            product_config_grid = list(product(*all_config_grids))
            for item in product_config_grid:
                base = dict(item[0])
                for sub_nr in range(1, len(item)):
                    base.update(item[sub_nr])
                self._config_grid.append(base)

    @property
    def config_grid(self):
        return self._config_grid

    def get_params(self, deep=True):
        all_params = {}
        for name, element in self.pipe_elements.items():
            all_params[name] = element.get_params(deep)
        return all_params

    def set_params(self, **kwargs):
        # Todo: disable fusion element?
        spread_params_dict = {}
        for k, val in kwargs.items():
            splitted_k = k.split('__')
            item_name = splitted_k[0]
            if item_name not in spread_params_dict:
                spread_params_dict[item_name] = {}
            dict_entry = {'__'.join(splitted_k[1::]): val}
            spread_params_dict[item_name].update(dict_entry)

        for name, params in spread_params_dict.items():
            if name in self.pipe_elements:
                self.pipe_elements[name].set_params(**params)
            else:
                Logger().error('NameError: Could not find element ' + name)
                raise NameError('Could not find element ', name)
        return self

    def fit(self, data, targets=None):
        for name, element in self.pipe_elements.items():
            # Todo: parallellize fitting
            element.fit(data, targets)
        return self

    def predict(self, data, targets=None):
        # Todo: strategy for concatenating data from different pipes
        # todo: parallelize prediction
        predicted_data = np.empty((0, 0))
        for name, element in self.pipe_elements.items():
            element_transform = element.predict(data)
            predicted_data = PipelineStacking.stack_data(predicted_data, element_transform)
        if self.voting:
            if hasattr(predicted_data, 'shape'):
                if len(predicted_data.shape) > 1:
                    predicted_data = np.mean(predicted_data, axis=1).astype(int)
        return predicted_data

    def predict_proba(self, data, targets=None):
        """
        Predict probabilities for every pipe element and
        stack them together. Alternatively, do voting instead.
        :param data: array-like
        :type data: float
        :param targets:
        :return: predicted values, array
        """
        # ToDo: Ask Ramona about "targets=None". Necessary?
        predicted_data = np.empty((0, 0))
        for name, element in self.pipe_elements.items():
            element_transform = element.predict_proba(data)
            predicted_data = PipelineStacking.stack_data(predicted_data, element_transform)
        if self.voting:
            if hasattr(predicted_data, 'shape'):
                if len(predicted_data.shape) > 1:
                    predicted_data = np.mean(predicted_data, axis=1).astype(int)
        return predicted_data

    def transform(self, data, targets=None):
        transformed_data = np.empty((0, 0))
        for name, element in self.pipe_elements.items():
            # if it is a hyperpipe with a final estimator, we want to use predict:
            if hasattr(element, 'pipe'):
                if element.overwrite_x is not None:
                    element_data = element.overwrite_x
                else:
                    element_data = data
                if element.pipe._final_estimator:
                    element_transform = element.predict(element_data)
                else:
                    # if it is just a preprocessing pipe we want to use transform
                    element_transform = element.transform(element_data)
            else:
                raise "I dont know what todo!"

            transformed_data = PipelineStacking.stack_data(transformed_data, element_transform)

        return transformed_data

    # def fit_predict(self, data, targets):
    #     predicted_data = None
    #     for name, element in self.pipe_elements.items():
    #         element_transform = element.fit_predict(data)
    #         predicted_data = PipelineStacking.stack_data(predicted_data, element_transform)
    #     return predicted_data
    #
    # def fit_transform(self, data, targets=None):
    #     transformed_data = np.empty((0, 0))
    #     for name, element in self.pipe_elements.items():
    #         # if it is a hyperpipe with a final estimator, we want to use predict:
    #         if hasattr(element, 'pipe'):
    #             if element.pipe._final_estimator:
    #                 element.fit(data, targets)
    #                 element_transform = element.predict(data)
    #             else:
    #                 # if it is just a preprocessing pipe we want to use transform
    #                 element.fit(data)
    #                 element_transform = element.transform(data)
    #             transformed_data = PipelineStacking.stack_data(transformed_data, element_transform)
    #     return transformed_data

    @classmethod
    def stack_data(cls, a, b):
        if not a.any():
            a = b
        else:
            # Todo: check for right dimensions!
            if a.ndim == 1 and b.ndim == 1:
                a = np.column_stack((a, b))
            else:
                b = np.reshape(b, (b.shape[0], 1))
                a = np.concatenate((a, b), 1)
        return a

    def score(self, X_test, y_test):
        # Todo: invent strategy for this ?
        # raise BaseException('PipelineStacking.score should probably never be reached.')
        # return 16
        predicted = self.predict(X_test)

        return accuracy_score(y_test, predicted)


class PipelineSwitch(PipelineElement):
    # @classmethod
    # def create(cls, pipeline_element_list):
    #     obj = PipelineSwitch()
    #     obj.pipeline_element_list = pipeline_element_list
    #     return obj

    def __init__(self, name, pipeline_element_list, _estimator_type='regressor'):
        self.name = name
        self._sklearn_curr_element = self.name + '__current_element'
        # Todo: disable switch?
        self.disabled = False
        self.set_disabled = False
        self._hyperparameters = {}
        self._sklearn_hyperparams = {}
        self.hyperparameters = self._hyperparameters
        self._config_grid = []
        self._current_element = (1, 1)
        self.pipeline_element_list = pipeline_element_list
        self.pipeline_element_configurations = []
        self.generate_config_grid()
        self.generate_sklearn_hyperparameters()
        self._estimator_type = _estimator_type

    @property
    def hyperparameters(self):
        # Todo: return actual hyperparameters of all pipeline elements??
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_config_grid(self):
        hyperparameters = []
        for i, pipe_element in enumerate(self.pipeline_element_list):
            element_configurations = []
            for item in pipe_element.config_grid:
                element_configurations.append(item)
            self.pipeline_element_configurations.append(element_configurations)
            hyperparameters += [(i, nr) for nr in range(len(element_configurations))]
        self._config_grid = [{self._sklearn_curr_element: (i, nr)} for i, nr in hyperparameters]
        self._hyperparameters = {'current_element': hyperparameters}

    @property
    def current_element(self):
        return self._current_element

    @property
    def config_grid(self):
        return self._config_grid

    @current_element.setter
    def current_element(self, value):
        self._current_element = value
        # pass the right config to the element
        # config = self.pipeline_element_configurations[value[0]][value[1]]
        # self.base_element.set_params(config)

    @property
    def base_element(self):
        obj = self.pipeline_element_list[self.current_element[0]]
        return obj

    def set_params(self, **kwargs):
        config_nr = None
        if self._sklearn_curr_element in kwargs:
            config_nr = kwargs[self._sklearn_curr_element]
        elif 'current_element' in kwargs:
            config_nr = kwargs['current_element']
        if config_nr is None or not isinstance(config_nr, (tuple, list)):
            Logger().error('ValueError: current_element must be of type Tuple')
            raise ValueError('current_element must be of type Tuple')
        else:
            self.current_element = config_nr
            config = self.pipeline_element_configurations[config_nr[0]][config_nr[1]]
            # remove name
            unnamed_config = {}
            for config_key, config_value in config.items():
                key_split = config_key.split('__')
                unnamed_config[''.join(key_split[1::])] = config_value
            self.base_element.set_params(**unnamed_config)

    def prettify_config_output(self, config_name, config_value, return_dict=False):
        if isinstance(config_value, tuple):
            output = self.pipeline_element_configurations[config_value[0]][config_value[1]]
            if not output:
                if return_dict:
                    return {self.pipeline_element_list[config_value[0]].name:None}
                else:
                    return self.pipeline_element_list[config_value[0]].name
            else:
                if return_dict:
                    return output
                return str(output)
        else:
            return super(PipelineSwitch, self).prettify_config_output(config_name, config_value)
