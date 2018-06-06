import pickle
import time
import datetime
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha1
from itertools import product
from copy import deepcopy
from collections import OrderedDict
import zipfile
import pickle
import glob
import inspect
import importlib.util
import os

from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from pymodm import connect

from photonai.configuration.Register import PhotonRegister
# from framework.ImbalancedWrapper import ImbalancedDataTransform
from photonai.photonlogger.Logger import Logger
from photonai.helpers.ConfigGrid import create_global_config
from photonai.optimization.OptimizationStrategies import GridSearchOptimizer, RandomGridSearchOptimizer, \
    TimeBoxedRandomGridSearchOptimizer
from photonai.validation.Validate import TestPipeline, OptimizerMetric
from photonai.validation.ResultsDatabase import *


class Hyperpipe(BaseEstimator):
    """
    Wrapper class for machine learning pipeline, holding all pipeline elements
    and managing the optimization of the hyperparameters

    Parameters
    ----------
    * 'name' [str]:
        Name of hyperpipe instance

    * 'inner_cv' [BaseCrossValidator]:
        Cross validation strategy to test hyperparameter configurations, generates the validation set

    * 'outer_cv' [BaseCrossValidator]:
        Cross validation strategy to use for the hyperparameter search itself, generates the test set

    * 'optimizer' [str or object, default="grid_search"]:
        Hyperparameter optimization algorithm

        - In case a string literal is given:
            - "grid_search": optimizer that iteratively tests all possible hyperparameter combinations
            - "random_grid_search": a variation of the grid search optimization that randomly picks hyperparameter
               combinations from all possible hyperparameter combinations
            - "timeboxed_random_grid_search": randomly chooses hyperparameter combinations from the set of all
               possible hyperparameter combinations and tests until the given time limit is reached
               - 'limit_in_minutes': int

        - In case an object is given:
          expects the object to have the following methods:
           - 'next_config_generator': returns a hyperparameter configuration in form of an dictionary containing
              key->value pairs in the sklearn parameter encoding 'model_name__parameter_name: parameter_value'
           - 'prepare': takes a list of pipeline elements and their particular hyperparameters to test
           - 'evaluate_recent_performance': gets a tested config and the respective performance in order to
              calculate a smart next configuration to process

    * 'metrics' [list of metric names as str]:
        Metrics that should be calculated for both training, validation and test set
        Use the preimported metrics from sklearn and photonai, or register your own

        - Metrics for 'classification':
            - 'accuracy': sklearn.metrics.accuracy_score
            - 'matthews_corrcoef': sklearn.metrics.matthews_corrcoef
            - 'confusion_matrix': sklearn.metrics.confusion_matrix,
            - 'f1_score': sklearn.metrics.f1_score
            - 'hamming_loss': sklearn.metrics.hamming_loss
            - 'log_loss': sklearn.metrics.log_loss
            - 'precision': sklearn.metrics.precision_score
            - 'recall': sklearn.metrics.recall_score
        - Metrics for 'regression':
            - 'mean_squared_error': sklearn.metrics.mean_squared_error
            - 'mean_absolute_error': sklearn.metrics.mean_absolute_error
            - 'explained_variance': sklearn.metrics.explained_variance_score
            - 'r2': sklearn.metrics.r2_score
        - Other metrics
            - 'pearson_correlation': photon_core.framework.Metrics.pearson_correlation
            - 'variance_explained':  photon_core.framework.Metrics.variance_explained_score
            - 'categorical_accuracy': photon_core.framework.Metrics.categorical_accuracy_score

    * 'best_config_metric' [str]:
        The metric that should be maximized or minimized in order to choose the best hyperparameter configuration

    * 'eval_final_performance' [bool, default=True]:
        If the metrics should be calculated for the test set, otherwise the test set is seperated but not used

    * 'test_size' [float, default=0.2]:
        the amount of the data that should be left out if no outer_cv is given and
        eval_final_perfomance is set to True

    * 'set_random_seed' [bool, default=False]:
        If True sets the random seed to 42

    * 'verbose' [int, default=0]:
        The level of verbosity, 0 is least talkative

    * 'logging' [bool, default=False]:
        If True, prints the output to a log file

    * 'logfile' [str]:
        Path to a file in which the log messages are written

    * 'groups' [array-like, default=None]:
        Info for advanced cross validation strategies, such as LeaveOneSiteOut-CV about the affiliation
        of the rows in the data

    * 'filter_element' [SourceFilter, default=None]:
        Instance of SourceFilter Class that transforms the input data, e.g. extracts certain columns

    * 'imbalanced_data_strategy_filter' [str, default=None]:
        Uses the imblearn package to handle imbalanced class distributions in the data
        A strategy is used to transform the data into more balanced distributions before the hyperparameter search
        is started.
        Strategies to choose from are:
        - imbalance_type = OVERSAMPLING:
            - RandomOverSampler
            - SMOTE
            - ADASYN

        -imbalance_type = UNDERSAMPLING:
            - ClusterCentroids,
            - RandomUnderSampler,
            - NearMiss,
            - InstanceHardnessThreshold,
            - CondensedNearestNeighbour,
            - EditedNearestNeighbours,
            - RepeatedEditedNearestNeighbours,
            - AllKNN,
            - NeighbourhoodCleaningRule,
            - OneSidedSelection

        - imbalance_type = COMBINE:
            - SMOTEENN,
            - SMOTETomek

    * 'config' [dict, default=None]:
        If the parameter is set, it constructs the pipeline elements from the information given in the dictionary.
        Can be used as a shortcut to add the pipeline elements.

    * 'debug_cv_mode' [bool, default=False]:
        Boolean for the internal unit tests of the nested cross validation

    Attributes
    ----------
    * 'optimum_pipe' [Pipeline]:
        An sklearn pipeline object that is fitted to the training data according to the best hyperparameter
        configuration found. Currently, we don't create an ensemble of all best hyperparameter configs over all folds.
        We find the best config by comparing the test error across outer folds. The hyperparameter config of the best
        fold is used as the optimal model and is then trained on the complete set.

    * 'best_config' [dict]:
        Dictionary containing the hyperparameters of the best configuration.
        Contains the parameters in the sklearn interface of model_name__parameter_name: parameter value

    * 'result_tree' [MDBHyperpipe]:
        Object containing all information about the for the performed hyperparameter search.
        Holds the training and test metrics for all outer folds, inner folds and configurations, as well as
        additional information.

    * 'pipeline_elements' [list]:
        Contains all PipelineElement or Hyperpipe objects that are added to the pipeline.

    Example
    -------
        manager = Hyperpipe('test_manager',
                            optimizer='timeboxed_random_grid_search', optimizer_params={'limit_in_minutes': 1},
                            outer_cv=ShuffleSplit(test_size=0.2, n_splits=1),
                            inner_cv=KFold(n_splits=10, shuffle=True),
                            metrics=['accuracy', 'precision', 'recall', "f1_score"],
                            best_config_metric='accuracy', eval_final_performance=True,
                            logging=True, verbose=2)

   """

    OPTIMIZER_DICTIONARY = {'grid_search': GridSearchOptimizer,
                            'random_grid_search': RandomGridSearchOptimizer,
                            'timeboxed_random_grid_search': TimeBoxedRandomGridSearchOptimizer}

    def __init__(self, name, inner_cv: BaseCrossValidator,
                 optimizer='grid_search', optimizer_params: dict = {}, groups=None, config=None, metrics=None,
                 best_config_metric=None, outer_cv=None,
                 test_size: float = 0.2, eval_final_performance=True,
                 calculate_metrics_per_fold: bool = True, calculate_metrics_across_folds: bool = False,
                 set_random_seed: bool=False,  filter_element=None, imbalanced_data_strategy_filter: str = None,
                 logfile: str = '', logging: bool =False, verbose: int =0, write_to_db: bool = True,
                 mongodb_connect_url: str = None, save_all_predictions: bool = False):


        # Re eval_final_performance:
        # set eval_final_performance to False because
        # 1. if no cv-object is given, no split is performed --> seems more logical
        #    than passing nothing, passing no cv-object but getting
        #    an 80/20 split by default
        # 2. if cv-object is given, split is performed but we don't peek
        #    into the test set --> thus we can evaluate more hp configs
        #    later without double dipping

        self.name = name
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.eval_final_performance = eval_final_performance
        self.test_size = test_size
        self.cv_iter = None
        self.data_test_cases = None

        self.calculate_metrics_per_fold = calculate_metrics_per_fold
        self.calculate_metrics_across_folds = calculate_metrics_across_folds

        # Todo: if self.outer_cv is LeaveOneOut: Set calculate metrics across folds to True -> Print

        self.X = None
        self.y = None

        self.groups = groups
        self.filter_element = filter_element
        # self.imbalanced_data_strategy_filter = ImbalancedDataTransform(imbalanced_data_strategy_filter)
        self.imbalanced_data_strategy_filter = None



        self.save_all_predictions = save_all_predictions

        self.fit_duration = 0

        self.logging = logging
        if set_random_seed:
            import random
            random.seed(42)
            print('set random seed to 42')
        self.verbose = verbose
        Logger().set_verbosity(self.verbose)
        if logfile:
            Logger().set_custom_log_file(logfile)

        # MongoDBWriter setup
        self.mongodb_connect_url = mongodb_connect_url
        self.write_to_db = write_to_db
        self.mongodb_writer = MongoDBWriter(write_to_db=write_to_db, connect_url=self.mongodb_connect_url)

        self.pipeline_elements = []
        self._pipe = None
        self.optimum_pipe = None

        self.metrics = metrics
        self.best_config_metric = best_config_metric
        self.config_optimizer = None

        self.result_tree = None
        self.best_config = None
        self.best_children_config = None
        self.best_performance = None
        self.is_final_fit = False

        self.__mother_outer_fold_counter = 0
        self.__mother_inner_fold_counter = 0
        self.__mother_config_counter = 0

        # containers for optimization history and logging
        self._performance_history_list = []

        if isinstance(config, dict):
            self.create_pipeline_elements_from_config(config)

        if isinstance(optimizer, str):
            # instantiate optimizer from string
            #  Todo: check if optimizer strategy is already implemented
            optimizer_class = self.OPTIMIZER_DICTIONARY[optimizer]
            optimizer_instance = optimizer_class(**optimizer_params)
            self.optimizer = optimizer_instance
        else:
            # Todo: check if correct object
            self.optimizer = optimizer

        self._validation_X = None
        self._validation_y = None
        self._test_X = None
        self._test_y = None
        self._last_fit_data_hash = None
        self._current_fold = -1
        self._num_of_folds = 0
        self._is_mother_pipe = True
        self._fold_data_hashes = []

    def __iadd__(self, pipe_element):
        """
        Add an element to the machine learning pipeline
        Returns self

        Parameters
        ----------
        * 'pipe_element' [PipelineElement or Hyperpipe]:
            The object to add to the machine learning pipeline, being either a transformer or an estimator.

        """
        # if isinstance(pipe_element, PipelineElement):
        self.pipeline_elements.append(pipe_element)
        # Todo: is repeated each time element is added....
        self._prepare_pipeline()
        return self
        # else:
        #     Todo: raise error
        # raise TypeError("Element must be of type Pipeline Element")

    def add(self, pipe_element):
        self.__iadd__(pipe_element)

    def __yield_all_data(self):
        """
        Helper function that iteratively returns the data stored in self.X
        Returns an iterable version of self.X
        """
        if hasattr(self.X, 'shape'):
            yield list(range(self.X.shape[0])), []
        else:
            yield list(range(len(self.X))), []

    def _generate_outer_cv_indices(self):
        """
        Generates the training and  test set indices for the hyperparameter search
        Returns a tuple of training and test indices

        - If there is a strategy given for the outer cross validation the strategy is called to split the data
        - If no strategy is given and eval_final_performance is True, all data is used for training
        - If no strategy is given and eval_final_performance is False: a test set is seperated from the
          training and validation set by the parameter test_size with ShuffleSplit
        """
        # if there is a CV Object for cross validating the hyperparameter search
        if self.outer_cv:
            self.data_test_cases = self.outer_cv.split(self.X, self.y)
        # in case we do not want to divide between validation and test set
        elif not self.eval_final_performance:
            self.data_test_cases = self.__yield_all_data()
        # the default is dividing one time into a validation and test set
        else:
            train_test_cv_object = ShuffleSplit(n_splits=1, test_size=self.test_size)
            self.data_test_cases = train_test_cv_object.split(self.X, self.y)

    def _distribute_cv_info_to_hyperpipe_children(self, reset: bool =False, reset_final_fit: bool=False,
                                                  outer_fold_counter: int=None, inner_fold_counter: int =None,
                                                  num_of_folds: int = None, config_counter: int =None):
        """
        Informs all elements of the pipeline that are of type hyperpipe (hyperpipe children)
        about the mother's configuration or current state

        Parameters
        ----------
        * 'num_of_folds' [int]:
            how many inner folds the mother hyperpipe has

        * 'outer_fold_counter' [int]:
            in which outer fold the mother hyerpipe currently is

        * 'inner_fold_counter' [int]:
            in which inner fold the mother hyperpipe currently is

        * 'config_counter' [int]:
            in which config_nr the mother hyperpipe actually is

        * 'reset' [bool, default = False]:
            if the hyperparameter search starts anew

        * 'reset_final_fit' [bool, default = False]:
            reset the is_final_fit parameter so that children hyperpipe train anew for outer fold of mother pipe

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

        # walk through all children of pipeline, if its a hyperpipe distribute the information
        for element_tuple in self._pipe.steps:
            element_object = element_tuple[1]
            if isinstance(element_object, Hyperpipe):
                _distribute_info_to_object(element_object, num_of_folds, reset, reset_final_fit,
                                          outer_fold_counter, inner_fold_counter, config_counter)
            elif isinstance(element_object, PipelineStacking):
                for child_pipe_name, child_pipe_object in element_object.pipe_elements.items():
                    if isinstance(child_pipe_object, Hyperpipe):
                        _distribute_info_to_object(child_pipe_object, num_of_folds, reset, reset_final_fit,
                                                  outer_fold_counter, inner_fold_counter, config_counter)

    def update_mother_inner_fold_nr(self, new_inner_fold_nr: int):
        """
        Function handle so that the TestPipeline class from Photon's validation module can pass the information to hyperpipe children

        Parameters
        ----------
        * 'new_inner_fold_nr' [int]:
            in which inner_fold the mother hyperpipe currently is
        """
        self._distribute_cv_info_to_hyperpipe_children(inner_fold_counter=new_inner_fold_nr)

    def fit(self, data, targets, **fit_params):
        """
        Starts the hyperparameter search and/or fits the pipeline to the data and targets

        Manages the nested cross validated hyperparameter search:

        1. Filters the data according to filter strategy (1) and according to the imbalanced_data_strategy (2)
        2. requests new configurations from the hyperparameter search strategy, the optimizer,
        3. initializes the testing of a specific configuration,
        4. communicates the result to the optimizer,
        5. repeats 2-4 until optimizer delivers no more configurations to test
        6. finally searches for the best config in all tested configs,
        7. trains the pipeline with the best config and evaluates the performance on the test set

        Parameters
        ----------
         * `data` [array-like, shape=[N, D]]:
            the training and test data, where N is the number of samples and D is the number of features.

         * `targets` [array-like, shape=[N]]:
            the truth values, where N is the number of samples.


        Returns
        -------
         * 'self'
            Returns self

        """

        # in case we want to inject some data from outside the pipeline

        self.X = data
        self.y = targets


        # !!!!!!!!!!!!!!!! FIT ONLY IF DATA CHANGED !!!!!!!!!!!!!!!!!!!
        # -------------------------------------------------------------

        # in case we need to reduce the dimension of the data due to parallelity of the outer pipe, lets do it.
        if self.filter_element:
            self.X = self.filter_element.transform(self.X)

        # if the groups are imbalanced, and a strategy is chosen, apply it here
        if self.imbalanced_data_strategy_filter:
            self.imbalanced_data_strategy_filter.fit(self.X, self.y)
            self.X, self.y = self.imbalanced_data_strategy_filter.transform()

        self._current_fold += 1

        # be compatible to list of (image-) files
        if isinstance(self.X, list):
            self.X = np.asarray(self.X)
        if isinstance(self.y, list):
            self.y = np.asarray(self.y)
        #if not isinstance(self.X, np.ndarray): # and isinstance(self.X[0], str):
        #    self.X = np.asarray(self.X)

        # handle neuro Imge paths as data
        # ToDo: Need to check the DATA, not the img paths for neuro
        new_data_hash = sha1(np.asarray(self.X, order='C')).hexdigest()

        # fit
        # 1. if it is first time ever or
        # 2. the data did change for that fold or
        # 3. if it is the mother pipe (then number_of_folds = 0)
        if (len(self._fold_data_hashes) < self._num_of_folds) \
                or (self._num_of_folds > 0 and self._fold_data_hashes[self._current_fold] != new_data_hash) \
                or self._num_of_folds == 0:

            # save data hash for that fold
            if self._num_of_folds > 0:
                if len(self._fold_data_hashes) < self._num_of_folds:
                    self._fold_data_hashes.append(new_data_hash)
                else:
                    self._fold_data_hashes[self._current_fold] = new_data_hash

            # optimize: iterate through configs and save results
            if not self.is_final_fit:

                # first check if correct optimizer metric has been chosen
                # pass pipeline_elements so that OptimizerMetric can look for last
                # element and use the corresponding score method
                self.config_optimizer = OptimizerMetric(self.best_config_metric, self.pipeline_elements, self.metrics)
                self.metrics = self.config_optimizer.check_metrics()

                if 'score' in self.metrics:
                    Logger().warn('Attention: Scoring with default score function of estimator can slow down calculations!')

                # generate OUTER ! cross validation splits to iterate over
                self._generate_outer_cv_indices()

                outer_fold_counter = 0

                if not self._is_mother_pipe:
                    self.result_tree_name = self.name + '_outer_fold_' + str(self.__mother_outer_fold_counter)  \
                                            + '_inner_fold_' + str(self.__mother_inner_fold_counter)
                else:
                    self.result_tree_name = self.name

                # initialize result logging with hyperpipe class
                self.result_tree = MDBHyperpipe(name=self.result_tree_name)
                self.result_tree.outer_folds = []

                # loop over outer cross validation
                for train_indices, test_indices in self.data_test_cases:

                    # give the optimizer the chance to inform about elements
                    self.optimizer.prepare(self.pipeline_elements)

                    outer_fold_counter += 1
                    outer_fold_fit_start_time = time.time()

                    Logger().info('HYPERPARAMETER SEARCH OF {0}, Outer Cross validation Fold {1}'
                                  .format(self.name, outer_fold_counter))

                    t1 = time.time()

                    # Prepare Train and validation set data
                    self._validation_X = self.X[train_indices]
                    self._validation_y = self.y[train_indices]
                    self._test_X = self.X[test_indices]
                    self._test_y = self.y[test_indices]

                    # Prepare inner cross validation
                    cv_iter = list(self.inner_cv.split(self._validation_X, self._validation_y))
                    num_folds = len(cv_iter)
                    num_samples_train = len(self._validation_y)
                    num_samples_test = len(self._test_y)

                    # distribute number of folds to encapsulated child hyperpipes
                    self._distribute_cv_info_to_hyperpipe_children(num_of_folds=num_folds,
                                                                   outer_fold_counter=outer_fold_counter)

                    tested_config_counter = 0

                    # add outer fold info object to result tree
                    outer_fold = MDBOuterFold(fold_nr=outer_fold_counter)
                    outer_fold.tested_config_list = []
                    self.result_tree.outer_folds.append(outer_fold)

                    # do the optimizing
                    for current_config in self.optimizer.next_config:
                        self._distribute_cv_info_to_hyperpipe_children(reset=True, config_counter=tested_config_counter)
                        hp = TestPipeline(self._pipe, current_config, self.metrics, self.update_mother_inner_fold_nr)
                        Logger().debug('optimizing of:' + self.name)
                        Logger().debug(self._optimize_printing(current_config))
                        Logger().debug('calculating...')

                        # Test the configuration cross validated by inner_cv object
                        current_config_mdb = hp.calculate_cv_score(self._validation_X, self._validation_y, cv_iter,
                                                            save_predictions=self.save_all_predictions,
                                                            calculate_metrics_per_fold=self.calculate_metrics_per_fold,
                                                            calculate_metrics_across_folds=self.calculate_metrics_across_folds)

                        current_config_mdb.config_nr = tested_config_counter
                        current_config_mdb.config_dict = current_config
                        current_config_mdb.pipe_name = self.name
                        tested_config_counter += 1
                        current_config_mdb.human_readable_config = self.config_to_dict(current_config)

                        # save the configuration of all children pipelines
                        children_config = {}
                        children_config_ref_list = []
                        for pipe_step in self._pipe.steps:
                            item = pipe_step[1]
                            if isinstance(item, Hyperpipe):
                                if item.local_search and item.best_config is not None:
                                    children_config[item.name] = item.best_config
                            elif isinstance(item, PipelineStacking):
                                for subhyperpipe_name, hyperpipe in item.pipe_elements.items():
                                    if isinstance(hyperpipe, Hyperpipe):
                                        if hyperpipe.local_search and hyperpipe.best_config is not None:
                                            # special case: we need to access pipe over pipeline_stacking element
                                            children_config[item.name + '__' + subhyperpipe_name] = hyperpipe.best_config.config_dict
                                        # children_config_ref_list.append(hyperpipe.best_config_outer_fold._id)
                        specific_parameters = self._pipe.get_params()
                        #current_config_mdb.full_model_spec = specific_parameters

                        current_config_mdb.children_config_dict = children_config
                        current_config_mdb.children_config_ref = children_config_ref_list

                        Logger().verbose(self._optimize_printing(current_config))

                        if not current_config_mdb.config_failed:
                            # get optimizer_metric and forward to optimizer
                            # todo: also pass greater_is_better=True/False to optimizer
                            metric_train = MDBHelper.get_metric(current_config_mdb, FoldOperations.MEAN, self.config_optimizer.metric)
                            metric_test = MDBHelper.get_metric(current_config_mdb, FoldOperations.MEAN, self.config_optimizer.metric, train=False)
                            #
                            # if not metric_train or metric_test:
                            #     raise Exception("Config did not fail, but did not get any metrics either....!!?")
                            config_performance = (metric_train, metric_test)

                            # Print Result for config
                            Logger().debug('...done:')
                            Logger().verbose(self.config_optimizer.metric + str(config_performance))
                        else:
                             config_performance = (-1, -1)
                             # Print Result for config
                             Logger().debug('...failed:')
                             Logger().error(current_config_mdb.config_error)

                        # add config to result tree and do intermediate saving
                        self.result_tree.outer_folds[-1].tested_config_list.append(current_config_mdb)
                        # Todo: add try catch in case config cannot be written
                        self.mongodb_writer.save(self.result_tree)

                        # 3. inform optimizer about performance
                        self.optimizer.evaluate_recent_performance(current_config, config_performance)

                    if tested_config_counter > 0:
                        best_config_outer_fold = self.config_optimizer.get_optimum_config(outer_fold.tested_config_list)

                        if not best_config_outer_fold:
                            raise Exception("No best config was found!")
                        best_config_outer_fold_mdb = MDBConfig()
                        best_config_outer_fold_mdb.children_config_dict = best_config_outer_fold.children_config_dict
                        best_config_outer_fold_mdb.pipe_name = self.name
                        best_config_outer_fold_mdb.children_config_ref = best_config_outer_fold.children_config_ref
                        # best_config_outer_fold_mdb.best_config_ref_to_train_item = best_config_outer_fold._id
                        best_config_outer_fold_mdb.config_dict = best_config_outer_fold.config_dict
                        best_config_outer_fold_mdb.human_readable_config = best_config_outer_fold.human_readable_config


                        # inform user
                        Logger().info('finished optimization of ' + self.name)
                        Logger().verbose('Result')
                        Logger().verbose('Number of tested configurations:' + str(tested_config_counter))
                        Logger().verbose('Optimizer metric: ' + self.config_optimizer.metric + '\n' +
                                         '   --> Greater is better: ' + str(self.config_optimizer.greater_is_better))
                        Logger().info('Best config: ' + self._optimize_printing(best_config_outer_fold_mdb.config_dict) +
                                      '\n' + '... with children config: '
                                      + self._optimize_printing(best_config_outer_fold_mdb.children_config_dict))


                        # ... and create optimal pipeline
                        self.optimum_pipe = self._pipe
                        # set self to best config
                        self.optimum_pipe.set_params(**best_config_outer_fold_mdb.config_dict)

                        # set all children to best config and inform to NOT optimize again, ONLY fit
                        for child_name, child_config in best_config_outer_fold_mdb.children_config_dict.items():
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

                        self._distribute_cv_info_to_hyperpipe_children(reset=True)

                        Logger().verbose('...now fitting ' + self.name + ' with optimum configuration')
                        fit_time_start = time.time()
                        self.optimum_pipe.fit(self._validation_X, self._validation_y)
                        final_fit_duration = time.time() - fit_time_start

                        #self.best_config_outer_fold.full_model_spec = self.optimum_pipe.get_params()
                        best_config_outer_fold_mdb.fit_duration_minutes = final_fit_duration
                        self.result_tree.outer_folds[-1].best_config = best_config_outer_fold_mdb
                        self.result_tree.outer_folds[-1].best_config.inner_folds = []

                        if self.eval_final_performance:
                            # Todo: generate mean and std over outer folds as well. move this items to the top
                            Logger().verbose('...now predicting ' + self.name + ' unseen data')

                            test_score_mdb = TestPipeline.score(self.optimum_pipe, self._test_X, self._test_y,
                                                                     self.metrics, save_predictions=self.save_all_predictions)

                            Logger().info('.. calculating metrics for test set (' + self.name + ')')
                            Logger().verbose('...now predicting ' + self.name + ' final model with training data')

                            train_score_mdb = TestPipeline.score(self.optimum_pipe, self._validation_X, self._validation_y,
                                                                      self.metrics, save_predictions=self.save_all_predictions)

                            # save test fold
                            outer_fold_mdb = MDBInnerFold()
                            outer_fold_mdb.fold_nr = 1
                            outer_fold_mdb.number_samples_training = num_samples_train
                            outer_fold_mdb.number_samples_validation = num_samples_test
                            outer_fold_mdb.training = train_score_mdb
                            outer_fold_mdb.validation = test_score_mdb
                            self.result_tree.outer_folds[-1].best_config.inner_folds = [outer_fold_mdb]


                            Logger().info('PERFORMANCE TRAIN:')
                            for m_key, m_value in train_score_mdb.metrics.items():
                                Logger().info(str(m_key) + ": " + str(m_value))

                            Logger().info('PERFORMANCE TEST:')
                            for m_key, m_value in test_score_mdb.metrics.items():
                                    Logger().info(str(m_key) + ": " + str(m_value))
                        else:

                            # save test fold
                            outer_fold_mdb = MDBInnerFold()
                            outer_fold_mdb.fold_nr = 1
                            outer_fold_mdb.number_samples_training = num_samples_train
                            outer_fold_mdb.number_samples_validation = num_samples_test

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
                            outer_fold_mdb.training = _copy_inner_fold_means(best_config_outer_fold.metrics_train)
                            # validation
                            outer_fold_mdb.validation = _copy_inner_fold_means(best_config_outer_fold.metrics_test)

                            self.result_tree.outer_folds[-1].best_config.inner_folds = [outer_fold_mdb]

                    Logger().info('This took {} minutes.'.format((time.time() - t1) / 60))
                    self.result_tree.time_of_results = datetime.datetime.now()
                    self.mongodb_writer.save(self.result_tree)
                    self._distribute_cv_info_to_hyperpipe_children(reset_final_fit=True, outer_fold_counter=outer_fold_counter)

                # Compute all final metrics
                self.result_tree.metrics_train, self.result_tree.metrics_test = MDBHelper.aggregate_metrics(self.result_tree.outer_folds,
                                                                                                            self.metrics)

                # Find best config across outer folds
                self.best_config = self.config_optimizer.get_optimum_config_outer_folds(self.result_tree.outer_folds)
                self.result_tree.best_config = self.best_config
                Logger().info('OVERALL BEST CONFIGURATION')
                Logger().info('--------------------------')
                Logger().info(self._optimize_printing(self.best_config.config_dict) +
                              '\n' + '... with children config: '
                              + self._optimize_printing(self.best_config.children_config_dict))
                # set self to best config
                self.optimum_pipe = self._pipe
                self.optimum_pipe.set_params(**self.best_config.config_dict)
                self.optimum_pipe.fit(self._validation_X, self._validation_y)

                # save results
                self.mongodb_writer.save(self.result_tree)
                if self.logging:
                    self.result_tree.print_csv_file(self.name + "_" + str(time.time()) + ".csv")
            ###############################################################################################
            else:
                self._pipe.fit(self.X, self.y, **fit_params)

        else:
            Logger().verbose("Avoided fitting of " + self.name + " on fold "
                             + str(self._current_fold) + " because data did not change")
            Logger().verbose('Best config of ' + self.name + ' : ' + str(self.best_config))

        return self

    def predict(self, data):
        """
        Use the optimum pipe to predict the data

        Returns
        -------
            predicted targets

        """
        # Todo: if local_search = true then use optimized pipe here?
        if self._pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.predict(data)

    def predict_proba(self, data):
        """
        Predict probabilities

       Returns
       -------

        """
        if self._pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.predict_proba(data)

    def transform(self, data):
        """
        Use the optimum pipe to transform the data
        :param data: the data to be transformed
        :type data: array-like
        :return: transformed data, array
        """
        if self._pipe:
            if self.filter_element:
                data = self.filter_element.transform(data)
            return self.optimum_pipe.transform(data)

    def get_params(self, deep=True):
        """
        Retrieve parameters from sklearn pipeline
        :param deep: If True, will return the parameters for this element and contained subobjects.
        :type deep: bool
        :return: dict of element_name__parameter_name: parameter_value
        """
        if self._pipe is not None:
            return self._pipe.get_params(deep)
        else:
            return None

    def set_params(self, **params):
        """
        Give parameter values to the pipeline elements
        :param params: dict of element_name__parameter_name: parameter_value
        :type params: dict
        :return: self
        """
        if self._pipe is not None:
            self._pipe.set_params(**params)
        return self

    def _prepare_pipeline(self):
        """
        build sklearn pipeline from PipelineElements and
        calculate parameter grid for all combinations of pipeline element hyperparameters
        """
        # prepare pipeline
        pipeline_steps = []
        for item in self.pipeline_elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))

        # build pipeline...
        self._pipe = Pipeline(pipeline_steps)

    def copy_me(self):
        """
        Helper function to copy all pipeline elements
        :return: list of copied pipeline elements
        """
        item_list =[]
        for item in self.pipeline_elements:
            item_list.append(item.copy_me())
        return item_list

    def _copy_pipeline(self):
        """
        Copy Pipeline by building a new sklearn Pipeline with Pipeline Elements
        :return: new sklearn Pipeline object
        """
        pipeline_steps = []
        for item in self.pipeline_elements:
            cpy = item.copy_me()
            if isinstance(cpy, list):
                for new_step in cpy:
                    pipeline_steps.append((new_step.name, new_step))
            else:
                pipeline_steps.append((cpy.name, cpy))
        return Pipeline(pipeline_steps)

    def save_optimum_pipe(self, file):
        """
        Save optimal pipeline only. Complete hyperpipe will no not be saved.
        :param: file: string specifying file to save pipeline to
        :return:
        """
        element_number = 0
        element_identifier = list()
        folder = os.path.splitext(file)[0]
        file = os.path.splitext(file)[0] + '.photon'

        if os.path.exists(folder):
            raise FileExistsError('Trying to save optimum pipe: The file you specified already exists as a folder.')
        else:
            os.mkdir(folder)
            folder = folder + '/'
        wrapper_files = list()

        for element_name, element in self.optimum_pipe.named_steps.items():
            filename = '_optimum_pipe_' + str(element_number) + '_' + element_name
            element_identifier.append({'element_name': element_name,
                                       'filename': filename})
            if hasattr(element.base_element, 'save'):
                element.base_element.save(folder + filename)
                element_identifier[-1]['mode'] = 'custom'
                element_identifier[-1]['wrapper_script'] = os.path.basename(inspect.getfile(element.base_element.__class__))
                wrapper_files.append(inspect.getfile(element.base_element.__class__))
                element_identifier[-1]['test_disabled'] = element.test_disabled
                element_identifier[-1]['disabled'] = element.disabled
                element_identifier[-1]['hyperparameters'] = element.hyperparameters

            else:
                try:
                    joblib.dump(element, folder + filename + '.pkl', compress=1)
                    element_identifier[-1]['mode'] = 'pickle'
                except:
                    raise NotImplementedError("Custom pipeline element must implement .save() method or "
                                              "allow pickle.")
            element_number += 1
        # save pipeline blueprint to make loading of pipeline easier
        with open(folder + '_optimum_pipe_blueprint.pkl', 'wb') as f:
            pickle.dump(element_identifier, f)

        # get all files
        files = glob.glob(folder + '_optimum_pipe_*')
        with zipfile.ZipFile(file, 'w') as myzip:
            for f in files:
                myzip.write(f, os.path.basename(f))
                os.remove(f)
            for f in wrapper_files:
                myzip.write(f, os.path.splitext(os.path.basename(f))[0] + '.py')
        os.removedirs(folder)

    @staticmethod
    def load_optimum_pipe(file):
        """
        Load optimal pipeline.
        :param file: string specifying .photon file to load optimal pipeline from
        :return:
        """
        if file.endswith('.photon'):
            archive_name = os.path.splitext(file)[0]
            folder = archive_name + '/'
            zf = zipfile.ZipFile(file)
            zf.extractall(folder)
        else:
            raise FileNotFoundError('Specify .photon file that holds PHOTON optimum pipe.')

        setup_info = pickle.load(open(folder + '_optimum_pipe_blueprint.pkl', 'rb'))
        element_list = list()
        for element_info in setup_info:
            if element_info['mode'] == 'custom':
                spec = importlib.util.spec_from_file_location(element_info['element_name'],
                                                              folder + element_info['wrapper_script'])
                imported_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imported_module)
                base_element = getattr(imported_module, element_info['element_name'])
                custom_element = PipelineElement(name=element_info['element_name'], base_element=base_element(),
                                                 hyperparameters=element_info['hyperparameters'],
                                                 test_disabled=element_info['test_disabled'],
                                                 disabled=element_info['disabled'])
                custom_element.base_element.load(folder + element_info['filename'])
                element_list.append((element_info['element_name'], custom_element))
            else:
                element_list.append((element_info['element_name'], joblib.load(folder + element_info['filename'] + '.pkl')))

        return Pipeline(element_list)


    def inverse_transform_pipeline(self, hyperparameters: dict, data, targets, data_to_inverse):
        """
        Inverse transform data for a pipeline with specific hyperparameter configuration

        1. Copy Sklearn Pipeline,
        2. Set Parameters
        3. Fit Pipeline to data and targets
        4. Inverse transform data with that pipeline
        :param hyperparameters: the settings for the pipeline elements
        :type hyperparameters: dict
        :param data: the training data
        :type data: array-like
        :param targets: the truth values for training
        :type targets: array-like
        :param data_to_inverse: the data that should be inversed after training
        :type data_to_inverse: array-like
        :return: inversed data as array
        """
        copied_pipe = self._copy_pipeline()
        copied_pipe.set_params(**hyperparameters)
        copied_pipe.fit(data, targets)
        return copied_pipe.inverse_transform(data_to_inverse)

    def _optimize_printing(self, config: dict):
        """
        make the sklearn config syntax prettily readable for humans
        :param config: a dict of pipeline_element_name__parameter_name: parameter_value
        :type config: dict
        :return: a prettified string containing all machines, parameters and values
        """
        prettified_config = [self.name + '\n']
        for el_key, el_value in config.items():
            items = el_key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in self._pipe.named_steps:
                new_pretty_key = '    ' + name + '->'
                prettified_config.append(new_pretty_key +
                                         self._pipe.named_steps[name].prettify_config_output(rest, el_value) + '\n')
            else:
                Logger().error('ValueError: Item is not contained in pipeline:' + name)
                raise ValueError('Item is not contained in pipeline:' + name)
        return ''.join(prettified_config)

    @staticmethod
    def prettify_config_output(config_name: str, config_value):
        """
        Print the disabled = False as Enabled = True for better human reading
        :param config_name: parameter name
        :type config_name: str
        :param config_value:  parameter value
        :return:
        """
        if config_name == "disabled" and config_value is False:
            return "enabled = True"
        else:
            return config_name + '=' + str(config_value)


    def create_pipeline_elements_from_config(self, config):
        """
        Create the pipeline from a config dict
        :param config:  dictionary of machine_name: hyperparameter dict
        """
        # Todo: Not reassign 'self'!!!
        for key, all_params in config.items():
            self += PipelineElement(key, all_params, {})

    def config_to_dict(self, specific_config):
        """
        :param specific_config:
        :return:
        """
        config = {}
        for key, value in specific_config.items():
            items = key.split('__')
            name = items[0]
            rest = '__'.join(items[1::])
            if name in self._pipe.named_steps:
                config.update(self._pipe.named_steps[name].prettify_config_output(rest, value, return_dict=True))
                #config[name] = value
        return config


class SourceFilter(BaseEstimator):
    """
    Helper Class to split the data e.g. for stacking.
    """
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Returns only part of the data, column-wise filtered by self.indices
        """
        return X[:, self.indices]


class PipelineElement(BaseEstimator):
    """
    Photon wrapper class for any transformer or predictor element in the pipeline.

    1. Saves the hyperparameters that are to be tested and creates a grid of all hyperparameter configurations
    2. Enables fast and rapid instantiation of pipeline elements per string identifier,
         e.g 'svc' creates an sklearn.svm.SVC object.
    3. Attaches a "disable" switch to every element in the pipeline in order to test a complete disable
    """
    # Registering Pipeline Elements
    ELEMENT_DICTIONARY = PhotonRegister.get_package_info(['PhotonCore', 'PhotonNeuro'])

    def __init__(self, name, hyperparameters: dict=None, test_disabled: bool=False, disabled:bool =False, base_element=None,
                 **kwargs):
        """
        Takes a string literal and transforms it into an object of the associated class (see PhotonCore.JSON)
        :param name: string literal encoding the class to be instantiated
        :type name: str
        :param hyperparameters: dict of parameter_name: [array of parameter values to be tested]
        :type hyperparameters: dict
        :param test_disabled: if the hyperparameter search should evaluate a complete disabling of the element
        :type  test_disabled: bool
        :param disabled: if the element is disabled
        :type disabled: bool
        :param kwargs: any parameter that should be passed to the object to be instantiiated, default parameters
        :return: instantiated class object
        """
        if hyperparameters is None:
            hyperparameters = {}

        if not base_element:
            if name in PipelineElement.ELEMENT_DICTIONARY:
                try:
                    desired_class_info = PipelineElement.ELEMENT_DICTIONARY[name]
                    desired_class_home = desired_class_info[0]
                    desired_class_name = desired_class_info[1]
                    imported_module = __import__(desired_class_home, globals(), locals(), desired_class_name, 0)
                    desired_class = getattr(imported_module, desired_class_name)
                    base_element = desired_class(**kwargs)
                    obj = PipelineElement(name, hyperparameters, test_disabled, disabled, base_element)
                    self.base_element = obj
                except AttributeError as ae:
                    Logger().error('ValueError: Could not find according class:'
                                   + str(PipelineElement.ELEMENT_DICTIONARY[name]))
                    raise ValueError('Could not find according class:', PipelineElement.ELEMENT_DICTIONARY[name])
            else:
                Logger().error('Element not supported right now:' + name)
                raise NameError('Element not supported right now:', name)
        else:
            self.base_element = base_element


        # Todo: check if hyperparameters are members of the class
        # Todo: write method that returns any hyperparameter that could be optimized --> sklearn: get_params.keys
        # Todo: map any hyperparameter to a possible default list of values to try
        self.name = name
        self.test_disabled = test_disabled
        self._sklearn_disabled = self.name + '__disabled'
        self._hyperparameters = hyperparameters
        # check if hyperparameters are already in sklearn style
        if len(hyperparameters) > 0:
            key_0 = next(iter(hyperparameters))
            if self.name not in key_0:
                self.hyperparameters = hyperparameters
        self.disabled = disabled

    def copy_me(self):
        return deepcopy(self)

    @classmethod
    def create(cls, name, base_element, hyperparameters: dict, test_disabled=False, disabled=False, **kwargs):
        return PipelineElement(name, hyperparameters, test_disabled, disabled, base_element=base_element, **kwargs)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: dict):
        self.generate_sklearn_hyperparameters(value)

    def generate_sklearn_hyperparameters(self, value: dict):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        :return: dict of hyperparameters
        """
        self._hyperparameters = {}
        for attribute, value_list in value.items():
            self._hyperparameters[self.name + '__' + attribute] = value_list
        if self.test_disabled:
            self._hyperparameters[self._sklearn_disabled] = [True]


    def get_params(self, deep: bool=True):
        """
        Forwards the get_params request to the wrapped base element
        :param deep: If True, will return the parameters for this element and contained subobjects.
        :type deep: bool
        :return: dict of params
        """
        return self.base_element.get_params(deep)

    def set_params(self, **kwargs):
        """
        Forwards the set_params request to the wrapped base element
        Takes care of the disabled parameter which is additionally attached by the PHOTON wrapper
        :param kwargs: the parameters to set
        :return: self
        """
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
        """
        Calls the fit function of the base element
        :param data: the data for fitting the element
        :type data: array-like
        :param targets: the targets for fitting the element
        :type targets: array
        :return: self
        """
        if not self.disabled:
            obj = self.base_element
            obj.fit(data, targets)
            # self.base_element.fit(data, targets)
        return self

    def predict(self, data):
        """
        Calls predict function on the base element.

        IF PREDICT IS NOT AVAILABLE CALLS TRANSFORM.
        This is for the case that the encapsulated hyperpipe only part of another hyperpipe, and works as a transformer.
        Sklearn usually expects the last element to predict.
        Also this is needed in case we are using an autoencoder which is firstly trained by using predict, and after
        training only used for transforming.

        :param data: the data on which the prediction should be based
        :type data: array-like
        :return:
        """
        if not self.disabled:
            if hasattr(self.base_element, 'predict'):
                return self.base_element.predict(data)
            elif hasattr(self.base_element, 'transform'):
                return self.base_element.transform(data)
            else:
                Logger().error('BaseException. base Element should have function ' +
                               'predict, or at least transform.')
                raise BaseException('base Element should have function predict, or at least transform.')
        else:
            return data

    def predict_proba(self, data):
        """
        Predict probabilities
        base element needs predict_proba() function, otherwise throw
        base exception.
        :param data: array-like
        :type data: float
        :return: predicted values, array
        """
        if not self.disabled:
            if hasattr(self.base_element, 'predict_proba'):
                return self.base_element.predict_proba(data)
            else:
                Logger().error('BaseException. base Element should have "predict_proba" function.')
            raise BaseException('base Element should have predict_proba function.')
        return data

    # def fit_predict(self, data, targets):
    #     if not self.disabled:
    #         return self.base_element.fit_predict(data, targets)
    #     else:
    #         return data

    def transform(self, data):
        """
        Calls transform on the base element.

        IN CASE THERE IS NO TRANSFORM METHOD, CALLS PREDICT.
        This is used if we are using an estimator as a preprocessing step.
        :param data: the data to transform
        :type data: array-like
        :return: array-like, the transformed data
        """
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
        """
        Calls inverse_transform on the base element
        :param data: the data to inverse
        :type data: array-like
        :return: array-like, the inversed data
        """
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
        """
        Calls the score function on the base element:
        Returns a goodness of fit measure or a likelihood of unseen data:

        :param X_test: data to score
        :param y_test: targets to be tested against
        :return: score_value (higher is better)
        """
        return self.base_element.score(X_test, y_test)

    def prettify_config_output(self, config_name: str, config_value, return_dict:bool=False):
        """Make hyperparameter combinations human readable """
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


class PipelineBranch(PipelineElement):
    """
     A substream of pipeline elements that is encapsulated e.g. for parallelization

     Parameters
     ----------
        * 'name' [str]:
            Name of the encapsulated item and/or summary of the encapsulated element's functions

        """

    def __init__(self, name):

        super().__init__(name, {}, test_disabled=False, disabled=False, base_element=True)

        self.pipeline_elements = []

    def __iadd__(self, pipe_element):
        """
        Add an element to the sub pipeline
        Returns self

        Parameters
        ----------
        * 'pipe_element' [PipelineElement or Hyperpipe]:
            The object to add, being either a transformer or an estimator.

        """
        self.pipeline_elements.append(pipe_element)
        self._prepare_pipeline()
        return self

    def add(self, pipe_element):
        self.__iadd__(pipe_element)

    def _prepare_pipeline(self):
        """ Generates sklearn pipeline with all underlying steps """
        pipeline_steps = []

        for item in self.pipeline_elements:
            # pipeline_steps.append((item.name, item.base_element))
            pipeline_steps.append((item.name, item))
            self._hyperparameters[item.name] = item.hyperparameters

        self.generate_sklearn_hyperparameters()
        self.base_element = Pipeline(pipeline_steps)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        return None

    def generate_sklearn_hyperparameters(self):
        """
        Generates a dictionary according to the sklearn convention of element_name__parameter_name: parameter_value
        :return: dict of hyperparameters
        """
        self._hyperparameters = {}
        for element in self.pipeline_elements:
            for attribute, value_list in element.hyperparameters.items():
                self._hyperparameters[self.name + '__' + attribute] = value_list


class PipelineStacking(PipelineElement):
    """
    Creates a vertical stacking/parallelization of pipeline items.

    The object acts as single pipeline element and encapsulates several vertically stacked other pipeline elements, each
    child receiving the same input data. The data is iteratively distributed to all children, the results are collected
    and horizontally concatenated.

    """
    def __init__(self, name: str, stacking_elements=None, voting: bool=True):
        """
        Creates a new PipelineStacking element.
        Collects all possible hyperparameter combinations of the children

        :param name: give the pipeline element a name
        :type name: str
        :param stacking_elements [list, optional]: list of pipeline elements that should run in parallel
        :param voting: if true, the predictions of the encapsulated pipeline elements are joined to a single prediction
        :type voting: bool
        """
        super(PipelineStacking, self).__init__(name, hyperparameters={}, test_disabled=False, disabled=False,
                                               base_element=True)

        self._hyperparameters = {}
        self.pipe_elements = OrderedDict()
        self.voting = voting
        if stacking_elements is not None:
            for item_to_stack in stacking_elements:
                self.__iadd__(item_to_stack)

    def __iadd__(self, item):
        """
        Adds a new element to the stack.
        Generates sklearn hyperparamter names in order to set the item's hyperparameters in the optimization process.

        * 'item' [PipelineElement or PipelineBranch or Hyperpipe]:
            The Element that should be stacked and will run in a vertical parallelization in the original pipe.
        """
        self.pipe_elements[item.name] = item
        self._hyperparameters[item.name] = item.hyperparameters

        # for each configuration
        tmp_dict = dict(item.hyperparameters)
        for key, element in tmp_dict.items():
            if isinstance(item, PipelineElement):
                self._hyperparameters[self.name + '__' + key] = tmp_dict[key]
            else:
                self._hyperparameters[self.name + '__' + item.name + '__' + key] = tmp_dict[key]
        return self

    def add(self, item):
        self.__iadd__(item)

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        """
        Setting hyperparameters does not make sense, only the items that added can be optimized, not the container (self)
        """
        pass

    def get_params(self, deep=True):
        all_params = {}
        for name, element in self.pipe_elements.items():
            all_params[name] = element.get_params(deep)
        return all_params

    def set_params(self, **kwargs):
        """
        Find the particular child and distribute the params to it
        :param kwargs: parameter dict of element_name__param_name: param_value
        :type kwargs: dict
        :return: self
        """
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
        """
        Calls fit iteratively on every child
        :param data: data for training
        :type data: array-like
        :param targets: accompanying targets for the training data
        :type targets: array-like
        :return: self
        """
        for name, element in self.pipe_elements.items():
            # Todo: parallellize fitting
            element.fit(data, targets)
        return self

    def predict(self, data):
        """
        Iteratively calls predict on every child.
        :param data: the data on which the predictions should be based
        :return: list of predictions
        """
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

    def predict_proba(self, data):
        """
        Predict probabilities for every pipe element and
        stack them together. Alternatively, do voting instead.
        :param data: array-like
        :type data: float
        :param targets:
        :return: predicted values, array
        """
        predicted_data = np.empty((0, 0))
        for name, element in self.pipe_elements.items():
            element_transform = element.predict_proba(data)
            predicted_data = PipelineStacking.stack_data(predicted_data, element_transform)
        if self.voting:
            if hasattr(predicted_data, 'shape'):
                if len(predicted_data.shape) > 1:
                    predicted_data = np.mean(predicted_data, axis=1).astype(int)
        return predicted_data

    def transform(self, data):
        """
        Calls transform on every child.

        If the encapsulated child is a hyperpipe, also calls predict on the last element in the pipeline.
        :param data: the data to transform
        :return: the data transformed by each child stacked
        """
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
        """
        Helper method to horizontally join the outcome of each child
        :param a: the existing matrix
        :param b: the matrix to attach horizontally
        :return: new matrix, that is a and b horizontally joined
        """
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
        """
        Calculate accuracy for predictions made with this object.
        This function should probably never be called.

        :param X_test:  data for the predictions
        :type X_test: array-like
        :param y_test: truth values
        :type y_test: list
        :return: accuracy_score
        """
        # Todo: invent strategy for this ?
        # raise BaseException('PipelineStacking.score should probably never be reached.')
        # return 16
        predicted = self.predict(X_test)

        return accuracy_score(y_test, predicted)


class PipelineSwitch(PipelineElement):
    """
    This class encapsulates several pipeline elements that belong at the same step of the pipeline,
    competing for being the best choice.

    If for example you want to find out if preprocessing A or preprocessing B is better at this position in the pipe.
    Or you want to test if a tree outperforms the good old SVM.

    ATTENTION: This class is a construct that may be convenient but is not suitable for any complex optimizations.
    Currently it only works for grid_search and the derived optimization strategies.
    USE THIS ONLY FOR RAPID PROTOTYPING AND PRELIMINARY RESULTS

    The class acts as if it is a single entity. Tt joins the hyperparamater combinations of each encapsulated element to
    a single, big combination grid. Each hyperparameter combination from that grid gets a number. Then the PipelineSwitch
    object publishes the numbers to be chosen as the object's hyperparameter. When a new number is chosen from the
    optimizer, it internally activates the belonging element and sets the element's parameter to the hyperparameter
    combination. In that way, each of the elements is tested in all its configurations at the same position in the
    pipeline. From the outside, the process and the optimizer only sees one parameter of the PipelineSwitch, that is
    the an integer indicating which item of the hyperparameter combination grid is currently active.

    """

    def __init__(self, name: str, pipeline_element_list: list = None, _estimator_type='regressor'):
        """
        Creates a new PipelineSwitch object and generated the hyperparameter combination grid
        :param name: how the element is called in the pipeline
        :param pipeline_element_list: the competing pipeline elements
        :param _estimator_type: used for validation purposes
        """
        self.name = name
        self.sklearn_name = self.name + "__current_element"
        self._hyperparameters = {}
        self._current_element = (1, 1)
        self.disabled = False
        self.pipeline_element_configurations = []
        self._estimator_type = _estimator_type

        if pipeline_element_list:
            self.pipeline_element_list = pipeline_element_list
            self.generate_config_grid()
        else:
            self.pipeline_element_list = []

    def __iadd__(self, other):
        self.pipeline_element_list.append(other)
        self.generate_config_grid()
        return self

    def add(self, other):
        self.__iadd__(other)

    @property
    def hyperparameters(self):
        # Todo: return actual hyperparameters of all pipeline elements??
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value):
        pass

    def generate_config_grid(self):
        # reset
        self.pipeline_element_configurations = []

        # calculate anew
        hyperparameters = []
        # generate possible combinations for each item respectively - do not mix hyperparameters across items
        for i, pipe_element in enumerate(self.pipeline_element_list):
            distinct_values_config = create_global_config([pipe_element])
            # add pipeline switch name in the config so that the hyperparameters can be set from other classes
            # pipeline switch will give the hyperparameters to the respective child
            distinct_values_config_copy = {}
            for config_key, config_value in distinct_values_config.items():
                distinct_values_config_copy[self.name + "__" + config_key] = config_value

            element_configurations = list(ParameterGrid(distinct_values_config_copy))
            self.pipeline_element_configurations.append(element_configurations)
            hyperparameters += [(i, nr) for nr in range(len(element_configurations))]

        self._hyperparameters = {self.sklearn_name: hyperparameters}

    @property
    def current_element(self):
        return self._current_element

    @current_element.setter
    def current_element(self, value):
        self._current_element = value
        # pass the right config to the element
        # config = self.pipeline_element_configurations[value[0]][value[1]]
        # self.base_element.set_params(config)

    @property
    def base_element(self):
        """
        Returns the currently active element
        :return: BaseTransformer or BaseEstimator object
        """
        obj = self.pipeline_element_list[self.current_element[0]]
        return obj

    def set_params(self, **kwargs):

        """
        The optimization process sees the amount of possible combinations and chooses one of them.
        Then this class activates the belonging element and prepared the element with the particular chosen configuration.
        :param kwargs: dict containing the current_element parameter


        Returns
        -------

        """

        config_nr = None
        if self.sklearn_name in kwargs:
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
                unnamed_config['__'.join(key_split[2::])] = config_value
            self.base_element.set_params(**unnamed_config)
        return self

    def prettify_config_output(self, config_name, config_value, return_dict=False):

        """
        Makes the sklearn configuration dictionary human readable

        :param config_name: the name of the parameter
        :param config_value: the value of the parameter
        :param return_dict: if True, the output is a dict with prettified keys

        Returns
        -------
        * 'prettified_configuration_string' [str]:
            configuration as prettified string or configuration as dict with prettified keys
        """

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
