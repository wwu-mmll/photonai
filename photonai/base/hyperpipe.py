import datetime
import importlib
import importlib.util
import inspect
import logging
import os
import pickle
import re
import shutil
import traceback
import zipfile
from copy import deepcopy
from typing import Optional, List, Union
import warnings

import dask
import numpy as np
import pandas as pd
from bson.objectid import ObjectId
from dask.distributed import Client
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
import joblib
from sklearn.model_selection._split import BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits
from sklearn.inspection import permutation_importance
from photonai.__init__ import __version__
from photonai.base.cache_manager import CacheManager
from photonai.base.photon_elements import Stack, Switch, Preprocessing, CallbackElement, Branch, PipelineElement, \
    PhotonNative
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.base.json_transformer import JsonTransformer
from photonai.optimization import FloatRange
from photonai.photonlogger.logger import logger
from photonai.processing import ResultsHandler
from photonai.optimization.optimization_info import Optimization
from photonai.processing.metrics import Scorer
from photonai.processing.outer_folds import OuterFoldManager
from photonai.processing.photon_folds import FoldInfo
from photonai.processing.results_structure import MDBHyperpipe, MDBHyperpipeInfo, MDBDummyResults, MDBHelper, \
     MDBConfig, MDBOuterFold


class OutputSettings:
    """
    Configuration class that specifies the format in which
    the results are saved. Results can be saved to a MongoDB
    or a simple son-file. You can also choose whether to save
    predictions and/or feature importances.
    """
    def __init__(self,
                 mongodb_connect_url: str = None,
                 save_output: bool = True,
                 overwrite_results: bool = False,
                 generate_best_model: bool = True,
                 user_id: str = '',
                 wizard_object_id: str = '',
                 wizard_project_name: str = '',
                 project_folder: str = ''):
        """
        Initialize the object.

        Parameters:
            mongodb_connect_url:
                Valid mongodb connection url that specifies a database for storing the results.

            save_output:
                Controls the general saving of the results.

            overwrite_results:
                Allows overwriting the results folder if it already exists.

            generate_best_model:
                Determines whether an optimum_pipe should be created and fitted.
                If False, no dependent files are created.

            user_id:
               The user name of the according PHOTONAI Wizard login.

            wizard_object_id:
               The object id to map the designed pipeline in the PHOTONAI Wizard
               to the results in the PHOTONAI CORE Database.

            wizard_project_name:
                How the project is titled in the PHOTONAI Wizard.

            project_folder:
                Deprecated Parameter - transferred to Hyperpipe.

        """
        if project_folder:
            msg = "Deprecated: The parameter 'project_folder' was moved to the Hyperpipe. " \
                  "Please use Hyperpipe(..., project_folder='')."
            logger.error(msg)
            raise DeprecationWarning(msg)
        self.mongodb_connect_url = mongodb_connect_url
        self.overwrite_results = overwrite_results

        self.user_id = user_id
        self.wizard_object_id = wizard_object_id
        self.wizard_project_name = wizard_project_name

        self.generate_best_model = generate_best_model
        self.save_output = save_output
        self.save_predictions_from_best_config_inner_folds = None

        self.verbosity = 0
        self.results_folder = ''
        self.project_folder = ''
        self.log_file = ''
        self.logging_file_handler = None

    # this is only allowed from hyperpipe
    def set_project_folder(self, project_folder):
        self.project_folder = project_folder
        self.initialize_log_file()

    @property
    def setup_error_file(self):
        if self.project_folder:
            return os.path.join(self.project_folder, 'photon_setup_errors.log')
        else:
            return ""

    def initialize_log_file(self):
        self.log_file = self.setup_error_file

    def update_settings(self, name, timestamp):

        if self.save_output:
            if not os.path.exists(self.project_folder):
                os.makedirs(self.project_folder)

            # Todo: give rights to user if this is done by docker container
            if self.overwrite_results:
                self.results_folder = os.path.join(self.project_folder, name + '_results')
            else:
                self.results_folder = os.path.join(self.project_folder, name + '_results_' + timestamp)

            logger.info("Output Folder: " + self.results_folder)

            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)

            if os.path.basename(self.log_file) == "photon_setup_errors.log":
                self.log_file = 'photon_output.log'
            self.log_file = self._add_timestamp(self.log_file)
            self.set_log_file()

        # if we made it here, there should be no further setup errors, every error that comes
        # now can go to the standard logger instance
        if os.path.isfile(self.setup_error_file):
            os.remove(self.setup_error_file)

    def _add_timestamp(self, file):
        return os.path.join(self.results_folder, os.path.basename(file))

    def _get_log_level(self):
        if self.verbosity == 0:
            level = 25
        elif self.verbosity == 1:
            level = logging.INFO  # 20
        elif self.verbosity == 2:
            level = logging.DEBUG  # 10
        else:
            level = logging.WARN  # 30
        return level

    def set_log_file(self):
        logfile_directory = os.path.dirname(self.log_file)
        if not os.path.exists(logfile_directory):
            os.makedirs(logfile_directory)
        if self.logging_file_handler is None:
            self.logging_file_handler = logging.FileHandler(self.log_file)
            self.logging_file_handler.setLevel(self._get_log_level())
            logger.addHandler(self.logging_file_handler)
        else:
            self.logging_file_handler.close()
            self.logging_file_handler.baseFilename = self.log_file

    def set_log_level(self):
        verbose_num = self._get_log_level()
        logger.setLevel(verbose_num)
        for handler in logger.handlers:
            handler.setLevel(verbose_num)


class Hyperpipe(BaseEstimator):
    """The PHOTONAI Hyperpipe class creates a custom
    machine learning pipeline. In addition it defines
    the relevant analysis’ parameters such as the
    cross-validation scheme, the hyperparameter optimization
    strategy, and the performance metrics of interest.

    So called PHOTONAI PipelineElements can be added to
    the Hyperpipe, each of them being a data-processing
    method or a learning algorithm. By choosing and
    combining data-processing methods or algorithms,
    and arranging them with the PHOTONAI classes,
    simple and complex pipeline architectures can be designed rapidly.

    The PHOTONAI Hyperpipe automatizes the nested training,
    test and hyperparameter optimization procedures.

    The Hyperpipe monitors:

    - the nested-cross-validated training
        and test procedure,
    - communicates with the hyperparameter optimization
        strategy,
    - streams information between the pipeline elements,
    - logs all results obtained and evaluates the performance,
    - guides the hyperparameter optimization process by
        a so-called best config metric which is used to select
        the best performing hyperparameter configuration.

    Attributes:
        optimum_pipe (PhotonPipeline):
            An sklearn pipeline object that is fitted to the training data
            according to the best hyperparameter configuration found.
            Currently, we don't create an ensemble of all best hyperparameter
            configs over all folds. We find the best config by comparing
            the test error across outer folds. The hyperparameter config of the best
            fold is used as the optimal model and is then trained on the complete set.

        best_config (dict):
            Dictionary containing the hyperparameters of the
            best configuration. Contains the parameters in the sklearn
            interface of model_name__parameter_name: parameter value.

        results (MDBHyperpipe):
            Object containing all information about the for the
            performed hyperparameter search. Holds the training and test
            metrics for all outer folds, inner folds
            and configurations, as well as additional information.

        elements (list):
            Contains `all PipelineElement or Hyperpipe
            objects that are added to the pipeline.

    Example:
        ``` python
        from photonai.base import Hyperpipe, PipelineElement
        from photonai.optimization import FloatRange
        from sklearn.model_selection import ShuffleSplit, KFold
        from sklearn.datasets import load_breast_cancer

        hyperpipe = Hyperpipe('myPipe',
                              optimizer='random_grid_search',
                              optimizer_params={'limit_in_minutes': 5},
                              outer_cv=ShuffleSplit(test_size=0.2, n_splits=3),
                              inner_cv=KFold(n_splits=10, shuffle=True),
                              metrics=['accuracy', 'precision', 'recall', "f1_score"],
                              best_config_metric='accuracy',
                              eval_final_performance=True,
                              verbosity=0)

        hyperpipe += PipelineElement("SVC", hyperparameters={"C": FloatRange(1, 100)})

        X, y = load_breast_cancer(return_X_y=True)
        hyperpipe.fit(X, y)
        ```

    """
    def __init__(self, name: Optional[str],
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits] = None,
                 outer_cv: Union[BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits, None] = None,
                 optimizer: str = 'grid_search',
                 optimizer_params: dict = None,
                 metrics: Optional[List[Union[Scorer.Metric_Type, str]]] = None,
                 best_config_metric: Optional[Union[Scorer.Metric_Type, str]] = None,
                 use_test_set: bool = True,
                 test_size: float = 0.2,
                 project_folder: str = '',
                 calculate_metrics_per_fold: bool = True,
                 calculate_metrics_across_folds: bool = False,
                 random_seed: int = None,
                 verbosity: int = 0,
                 learning_curves: bool = False,
                 learning_curves_cut: FloatRange = None,
                 output_settings: OutputSettings = None,
                 performance_constraints: list = None,
                 permutation_id: str = None,
                 cache_folder: str = None,
                 nr_of_processes: int = 1,
                 allow_multidim_targets: bool = False):
        """
        Initialize the object.

        Parameters:
            name:
                Name of hyperpipe instance.

            inner_cv:
                Cross validation strategy to test hyperparameter configurations, generates the validation set.

            outer_cv:
                Cross validation strategy to use for the hyperparameter search itself, generates the test set.

            optimizer:
                Hyperparameter optimization algorithm.

                - In case a string literal is given:
                    - "grid_search": Optimizer that iteratively tests all possible hyperparameter combinations.
                    - "random_grid_search": A variation of the grid search optimization that randomly picks
                        hyperparameter combinations from all possible hyperparameter combinations.
                    - "sk_opt": Scikit-Optimize based on theories of Baysian optimization.
                    - "random_search": randomly chooses hyperparameter from grid-free domain.
                    - "smac": SMAC based on theories of Baysian optimization.
                    - "nevergrad": Nevergrad based on theories of evolutionary learning.

                - In case an object is given:
                    expects the object to have the following methods:
                    - `ask`: returns a hyperparameter configuration in form of an dictionary containing
                        key->value pairs in the sklearn parameter encoding `model_name__parameter_name: parameter_value`
                    - `prepare`: takes a list of pipeline elements and their particular hyperparameters to prepare the
                                 hyperparameter space
                    - `tell`: gets a tested config and the respective performance in order to
                        calculate a smart next configuration to process

            metrics:
                Metrics that should be calculated for both training, validation and test set
                Use the preimported metrics from sklearn and photonai, or register your own

                - Metrics for `classification`:
                    - `accuracy`: sklearn.metrics.accuracy_score
                    - `matthews_corrcoef`: sklearn.metrics.matthews_corrcoef
                    - `confusion_matrix`: sklearn.metrics.confusion_matrix,
                    - `f1_score`: sklearn.metrics.f1_score
                    - `hamming_loss`: sklearn.metrics.hamming_loss
                    - `log_loss`: sklearn.metrics.log_loss
                    - `precision`: sklearn.metrics.precision_score
                    - `recall`: sklearn.metrics.recall_score
                - Metrics for `regression`:
                    - `mean_squared_error`: sklearn.metrics.mean_squared_error
                    - `mean_absolute_error`: sklearn.metrics.mean_absolute_error
                    - `explained_variance`: sklearn.metrics.explained_variance_score
                    - `r2`: sklearn.metrics.r2_score
                - Other metrics
                    - `pearson_correlation`: photon_core.framework.Metrics.pearson_correlation
                    - `variance_explained`:  photon_core.framework.Metrics.variance_explained_score
                    - `categorical_accuracy`: photon_core.framework.Metrics.categorical_accuracy_score

            best_config_metric:
                The metric that should be maximized or minimized in order to choose
                the best hyperparameter configuration.

            use_test_set [bool, default=True]:
                If the metrics should be calculated for the test set,
                otherwise the test set is seperated but not used.

            project_folder:
                The output folder in which all files generated by the
                PHOTONAI project are saved to.

            test_size:
                The amount of the data that should be left out if no outer_cv is given and
                eval_final_perfomance is set to True.

            calculate_metrics_per_fold:
                If True, the metrics are calculated for each inner_fold.
                If False, calculate_metrics_across_folds must be True.

            calculate_metrics_across_folds:
                If True, the metrics are calculated across all inner_fold.
                If False, calculate_metrics_per_fold must be True.

            random_seed:
                Random Seed.

            verbosity:
                The level of verbosity, 0 is least talkative and
                gives only warn and error, 1 gives adds info and 2 adds debug.

            learning_curves:
                Enables larning curve procedure. Evaluate learning process over
                different sizes of input. Depends on learning_curves_cut.

            learning_curves_cut:
                The tested relativ cuts for data size.

            performance_constraints:
                Objects that indicate whether a configuration should
                be tested further. For example, the inner fold of a config
                does not perform better than the dummy performance.

            permutation_id:
                String identifier for permutation tests.

            cache_folder:
                Folder path for multi-processing.

            nr_of_processes:
                Determined the amount of simultaneous calculation of outer_folds.

            allow_multidim_targets:
                Allows multidimensional targets.

        """

        self.name = re.sub(r'\W+', '', name)

        # ====================== Cross Validation ===========================
        # check if both calculate_metrics_per_folds and calculate_metrics_across_folds is False
        if not calculate_metrics_across_folds and not calculate_metrics_per_fold:
            raise NotImplementedError("Apparently, you've set calculate_metrics_across_folds=False and "
                                      "calculate_metrics_per_fold=False. In this case PHOTONAI does not calculate "
                                      "any metrics which doesn't make any sense. Set at least one to True.")
        if inner_cv is None:
            msg = "PHOTONAI requires an inner_cv split. Please enable inner cross-validation. " \
                  "As exmaple: Hyperpipe(...inner_cv = KFold(n_splits = 3), ...). " \
                  "Ensure you import the cross_validation object first."
            logger.error(msg)
            raise AttributeError(msg)

        # use default cut 'FloatRange(0, 1, 'range', 0.2)' if learning_curves = True but learning_curves_cut is None
        if learning_curves and learning_curves_cut is None:
            learning_curves_cut = FloatRange(0, 1, 'range', 0.2)
        elif not learning_curves and learning_curves_cut is not None:
            learning_curves_cut = None

        self.cross_validation = Hyperpipe.CrossValidation(inner_cv=inner_cv,
                                                          outer_cv=outer_cv,
                                                          use_test_set=use_test_set,
                                                          test_size=test_size,
                                                          calculate_metrics_per_fold=calculate_metrics_per_fold,
                                                          calculate_metrics_across_folds=calculate_metrics_across_folds,
                                                          learning_curves=learning_curves,
                                                          learning_curves_cut=learning_curves_cut)

        # ====================== Data ===========================
        self.data = Hyperpipe.Data()

        # ====================== Output Folder and Log File Management ===========================
        if output_settings:
            self.output_settings = output_settings
        else:
            self.output_settings = OutputSettings()

        if project_folder == '':
            self.project_folder = os.getcwd()
        else:
            self.project_folder = project_folder

        self.output_settings.set_project_folder(self.project_folder)

        # update output options to add pipe name and timestamp to results folder
        self._verbosity = 0
        self.verbosity = verbosity
        self.output_settings.set_log_file()

        # ====================== Result Logging ===========================
        self.results_handler = None
        self.results = None
        self.best_config = None

        # ====================== Pipeline ===========================
        self.elements = []
        self._pipe = None
        self.optimum_pipe = None
        self.preprocessing = None

        # ====================== Performance Optimization ===========================
        if optimizer_params is None:
            optimizer_params = {}
        self.optimization = Optimization(metrics=metrics,
                                         best_config_metric=best_config_metric,
                                         optimizer_input=optimizer,
                                         optimizer_params=optimizer_params,
                                         performance_constraints=performance_constraints)

        # self.optimization.sanity_check_metrics()

        # ====================== Caching and Parallelization ===========================
        self.nr_of_processes = nr_of_processes
        if cache_folder:
            self.cache_folder = os.path.join(cache_folder, self.name)
        else:
            self.cache_folder = None

        # ====================== Internals ===========================

        self.permutation_id = permutation_id
        self.allow_multidim_targets = allow_multidim_targets
        self.is_final_fit = False

        # ====================== Random Seed ===========================
        self.random_state = random_seed
        if random_seed is not None:
            import random
            random.seed(random_seed)

    # ===================================================================
    # Helper Classes
    # ===================================================================

    class CrossValidation:

        def __init__(self, inner_cv, outer_cv,
                     use_test_set, test_size,
                     calculate_metrics_per_fold,
                     calculate_metrics_across_folds,
                     learning_curves,
                     learning_curves_cut):
            self.inner_cv = inner_cv
            self.outer_cv = outer_cv
            self.use_test_set = use_test_set
            self.test_size = test_size

            self.learning_curves = learning_curves
            self.learning_curves_cut = learning_curves_cut

            self.calculate_metrics_per_fold = calculate_metrics_per_fold
            # Todo: if self.outer_cv is LeaveOneOut: Set calculate metrics across folds to True -> Print
            self.calculate_metrics_across_folds = calculate_metrics_across_folds

            self.outer_folds = None
            self.inner_folds = dict()

    def __str__(self):
        return "Hyperpipe {}".format(self.name)

    class Data:

        def __init__(self, X=None, y=None, kwargs=None, allow_multidim_targets=False):
            self.X = X
            self.y = y
            self.kwargs = kwargs
            self.allow_multidim_targets = allow_multidim_targets

        def input_data_sanity_checks(self, data, targets, **kwargs):
            # ==================== SANITY CHECKS ===============================
            # 1. Make to numpy arrays
            # 2. erase all Nan targets

            logger.info("Checking input data...")
            self.X = data
            self.y = targets
            self.kwargs = kwargs

            try:
                if self.X is None:
                    raise ValueError("(Input-)data is a NoneType.")
                if self.y is None:
                    raise ValueError("(Input-)target is a NoneType.")

                shape_x = np.shape(self.X)
                shape_y = np.shape(self.y)
                if not self.allow_multidim_targets:
                    if len(shape_y) != 1:
                        if len(np.shape(np.squeeze(self.y))) == 1:
                            # use np.squeeze for non 1D targets.
                            self.y = np.squeeze(self.y)
                            shape_y = np.shape(self.y)
                            msg = "y has been automatically squeezed. If this is not your intention, block this " \
                                  "with Hyperpipe(allow_multidim_targets = True"
                            logger.warning(msg)
                            warnings.warn(msg)
                        else:
                            raise ValueError(
                                "Target is not one-dimensional. Multidimensional targets can cause problems"
                                "with sklearn metrics. Please override with "
                                "Hyperpipe(allow_multidim_targets = True).")
                if not shape_x[0] == shape_y[0]:
                    raise IndexError(
                        "Size of targets mismatch to size of the data: " + str(shape_x[0]) + " - " + str(shape_y[0]))
            except IndexError as ie:
                logger.error("IndexError: " + str(ie))
                raise ie
            except ValueError as ve:
                logger.error("ValueError: " + str(ve))
                raise ve
            except Exception as e:
                logger.error("Error: " + str(e))
                raise e

            # be compatible to list of (image-) files
            if isinstance(self.X, list):
                self.X = np.asarray(self.X)
            elif isinstance(self.X, (pd.DataFrame, pd.Series)):
                self.X = self.X.to_numpy()
            if isinstance(self.y, list):
                self.y = np.asarray(self.y)
            elif isinstance(self.y, pd.Series) or isinstance(self.y, pd.DataFrame):
                self.y = self.y.to_numpy()

            # at first first, erase all rows where y is Nan if preprocessing has not done it already
            try:
                nans_in_y = np.isnan(self.y)
                nr_of_nans = len(np.where(nans_in_y == 1)[0])
                if nr_of_nans > 0:
                    logger.info("You have {} Nans in your target vector, "
                                "PHOTONAI erases every data item that has a Nan Target".format(str(nr_of_nans)))
                    self.X = self.X[~nans_in_y]
                    self.y = self.y[~nans_in_y]
            except Exception as e:
                # This is only for convenience so if it fails then never mind
                logger.error("Removing Nans in target vector failed: " + str(e))
                pass

            logger.info("Running analysis with " + str(self.y.shape[0]) + " samples.")

    # ===================================================================
    # Properties and Helper
    # ===================================================================
    @property
    def estimation_type(self):
        estimation_type = getattr(self.elements[-1], '_estimator_type')
        if estimation_type is None:
            raise NotImplementedError("Last element in Hyperpipe should be an estimator.")
        else:
            return estimation_type

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value
        self.output_settings.verbosity = self._verbosity
        self.output_settings.set_log_level()

    @staticmethod
    def disable_multiprocessing_recursively(pipe):
        if isinstance(pipe, (Stack, Branch, Switch, Preprocessing)):
            if hasattr(pipe, 'nr_of_processes'):
                pipe.nr_of_processes = 1
            for child in pipe.elements:
                if hasattr(child, 'base_element'):
                    Hyperpipe.disable_multiprocessing_recursively(child.base_element)
        elif isinstance(pipe, PhotonPipeline):
            for name, child in pipe.named_steps.items():
                Hyperpipe.disable_multiprocessing_recursively(child)
        else:
            if hasattr(pipe, 'nr_of_processes'):
                pipe.nr_of_processes = 1

    @staticmethod
    def recursive_cache_folder_propagation(element, cache_folder, inner_fold_id):
        if isinstance(element, (Switch, Stack, Preprocessing)):
            for child in element.elements:
                Hyperpipe.recursive_cache_folder_propagation(child, cache_folder, inner_fold_id)

        elif isinstance(element, Branch):
            # in case it's a Branch, we create a cache subfolder and propagate it to every child
            if cache_folder:
                cache_folder = os.path.join(cache_folder, element.name)
            Hyperpipe.recursive_cache_folder_propagation(element.base_element, cache_folder, inner_fold_id)
            # Hyperpipe.prepare_caching(element.base_element.cache_folder)

        elif isinstance(element, PhotonPipeline):
            element.fold_id = inner_fold_id
            element.cache_folder = cache_folder

            # pipe.caching is automatically set to True or False by .cache_folder setter

            for name, child in element.named_steps.items():
                # we need to check if any element is Branch, Stack or Swtich
                Hyperpipe.recursive_cache_folder_propagation(child, cache_folder, inner_fold_id)

        # else: if it's a simple PipelineElement, then we just don't do anything

    # ===================================================================
    # Pipeline Setup
    # ===================================================================

    def __iadd__(self, pipe_element: PipelineElement):
        """
        Add an element to the machine learning pipeline.
        Returns self.

        Parameters:
            pipe_element:
                The object to add to the machine learning pipeline,
                being either a transformer or an estimator.

        """
        if isinstance(pipe_element, Preprocessing):
            self.preprocessing = pipe_element
        elif isinstance(pipe_element, CallbackElement):
            pipe_element.needs_y = True
            self.elements.append(pipe_element)
        else:
            if isinstance(pipe_element, PipelineElement) or issubclass(type(pipe_element), PhotonNative):
                self.elements.append(pipe_element)
            else:
                raise TypeError("Element must be of type Pipeline Element")
        return self

    def add(self, pipe_element: PipelineElement):
        """
        Add an element to the machine learning pipeline.
        Returns self.

        Parameters:
            pipe_element:
                The object to add to the machine learning pipeline,
                being either a transformer or an estimator.

        """
        self.__iadd__(pipe_element)

    # ===================================================================
    # Workflow Setup
    # ===================================================================
    def _prepare_dummy_estimator(self):
        self.results.dummy_estimator = MDBDummyResults()

        if self.estimation_type == 'regressor':
            self.results.dummy_estimator.strategy = 'mean'
            return DummyRegressor(strategy=self.results.dummy_estimator.strategy)
        elif self.estimation_type == 'classifier':
            self.results.dummy_estimator.strategy = 'most_frequent'
            return DummyClassifier(strategy=self.results.dummy_estimator.strategy)
        else:
            logger.info('Estimator does not specify whether it is a regressor or classifier. '
                        'DummyEstimator step skipped.')
            return

    def __get_pipeline_structure(self, pipeline_elements):
        element_list = dict()
        for p_el in pipeline_elements:
            if not hasattr(p_el, 'name'):
                raise Warning('Strange Pipeline Element found that has no name..? Type: '.format(type(p_el)))
            if hasattr(p_el, 'elements'):
                child_list = self.__get_pipeline_structure(p_el.elements)
                identifier = p_el.name
                if hasattr(p_el, "identifier"):
                    identifier = p_el.identifier + identifier
                    element_list[identifier] = child_list
            else:
                if hasattr(p_el, 'base_element'):
                    element_list[p_el.name] = str(type(p_el.base_element))
                else:
                    element_list[p_el.name] = str(type(p_el))
        return element_list

    def _prepare_result_logging(self, start_time):

        self.results = MDBHyperpipe(name=self.name, version=__version__)
        self.results.hyperpipe_info = MDBHyperpipeInfo()

        # in case eval final performance is false, we have no outer fold predictions
        if not self.cross_validation.use_test_set:
            self.output_settings.save_predictions_from_best_config_inner_folds = True
        self.results_handler = ResultsHandler(self.results, self.output_settings)

        self.results.computation_start_time = start_time
        self.results.hyperpipe_info.estimation_type = self.estimation_type
        self.results.output_folder = self.output_settings.results_folder

        if self.permutation_id is not None:
            self.results.permutation_id = self.permutation_id

        # save wizard information to PHOTONAI db in order to map results to the wizard design object
        if self.output_settings and hasattr(self.output_settings, 'wizard_object_id'):
            if self.output_settings.wizard_object_id:
                self.name = self.output_settings.wizard_object_id
                self.results.name = self.output_settings.wizard_object_id
                self.results.wizard_object_id = ObjectId(self.output_settings.wizard_object_id)
                self.results.wizard_system_name = self.output_settings.wizard_project_name
                self.results.user_id = self.output_settings.user_id
        self.results.outer_folds = []
        self.results.hyperpipe_info.elements = self.__get_pipeline_structure(self.elements)
        self.results.hyperpipe_info.eval_final_performance = self.cross_validation.use_test_set
        self.results.hyperpipe_info.best_config_metric = self.optimization.best_config_metric
        self.results.hyperpipe_info.metrics = self.optimization.metrics
        self.results.hyperpipe_info.learning_curves_cut = self.cross_validation.learning_curves_cut
        self.results.hyperpipe_info.maximize_best_config_metric = self.optimization.maximize_metric

        # optimization
        def _format_cross_validation(cv):
            if cv:
                string = "{}(".format(cv.__class__.__name__)
                for key, val in cv.__dict__.items():
                    string += "{}={}, ".format(key, val)
                return string[:-2] + ")"
            else:
                return "None"

        self.results.hyperpipe_info.cross_validation = \
            {'OuterCV': _format_cross_validation(self.cross_validation.outer_cv),
             'InnerCV': _format_cross_validation(self.cross_validation.inner_cv)}
        self.results.hyperpipe_info.data = {'X_shape': self.data.X.shape, 'y_shape': self.data.y.shape}
        self.results.hyperpipe_info.optimization = {'Optimizer': self.optimization.optimizer_input_str,
                                                    'OptimizerParams': str(self.optimization.optimizer_params),
                                                    'BestConfigMetric': self.optimization.best_config_metric}

        # add json file of hyperpipe attributes
        try:
            json_transformer = JsonTransformer()
            json_transformer.to_json_file(self, self.output_settings.results_folder+"/hyperpipe_config.json")
        except:
            msg = "JsonTransformer was unable to create the .json file."
            logger.warning(msg)
            warnings.warn(msg)

    def _finalize_optimization(self):
        # ==================== EVALUATING RESULTS OF HYPERPARAMETER OPTIMIZATION ===============================
        # 1. computing average metrics
        # 2. finding overall best config
        # 3. training model with best config
        # 4. persisting best model
        logger.clean_info('')
        logger.stars()
        logger.photon_system_log("Finished all outer fold computations.")
        logger.info("Now analysing the final results...")

        # computer dummy metrics
        logger.info("Computing dummy metrics...")
        config_item = MDBConfig()
        dummy_results = [outer_fold.dummy_results for outer_fold in self.results.outer_folds]
        config_item.inner_folds = [f for f in dummy_results if f is not None]
        if len(config_item.inner_folds) > 0:
            self.results.dummy_estimator.metrics_train, self.results.dummy_estimator.metrics_test = \
                MDBHelper.aggregate_metrics_for_inner_folds(config_item.inner_folds, self.optimization.metrics)

        logger.info("Computing mean and std for all outer fold metrics...")
        # Compute all final metrics
        self.results.metrics_train, self.results.metrics_test = \
            MDBHelper.aggregate_metrics_for_outer_folds(self.results.outer_folds, self.optimization.metrics)

        # Find best config across outer folds
        logger.info("Find best config across outer folds...")
        best_config = self.optimization.get_optimum_config_outer_folds(self.results.outer_folds)
        self.best_config = best_config.config_dict
        self.results.best_config = best_config

        # save results again
        self.results.computation_end_time = datetime.datetime.now()
        self.results.computation_completed = True
        logger.info("Save final results...")
        self.results_handler.save()

        logger.info("Prepare Hyperpipe.optimum pipe with best config..")
        # set self to best config
        self.optimum_pipe = self._pipe
        self.optimum_pipe.set_params(**self.best_config)

        if self.output_settings.generate_best_model:
            logger.info("Fitting best model...")
            # set self to best config
            self.optimum_pipe = self._pipe
            self.optimum_pipe.set_params(**self.best_config)

            # set caching
            # we want caching disabled in general but still want to do single subject caching
            self.recursive_cache_folder_propagation(self.optimum_pipe, self.cache_folder, 'fixed_fold_id')
            self.optimum_pipe.caching = False

            # disable multiprocessing when fitting optimum pipe
            # (otherwise inverse_transform won't work for BrainAtlas/Mask)
            self.disable_multiprocessing_recursively(self.optimum_pipe)

            self.optimum_pipe.fit(self.data.X, self.data.y, **self.data.kwargs)

            # Before saving the optimum pipe, add preprocessing without multiprocessing
            self.optimum_pipe.add_preprocessing(self.disable_multiprocessing_recursively(self.preprocessing))

            # Now truly set to no caching (including single_subject_caching)
            self.recursive_cache_folder_propagation(self.optimum_pipe, None, None)

            if self.output_settings.save_output:
                try:
                    pretrained_model_filename = os.path.join(self.output_settings.results_folder,
                                                             'photon_best_model.photon')
                    PhotonModelPersistor.save_optimum_pipe(self.optimum_pipe, pretrained_model_filename)
                    logger.info("Saved best model to file.")
                except Exception as e:
                    logger.info("Could not save best model to file")
                    logger.error(str(e))

                # get feature importances of optimum pipe
                logger.info("Mapping back feature importances...")
                feature_importances = self.optimum_pipe.feature_importances_

                if not feature_importances:
                    logger.info("No feature importances available for {}!".format(self.optimum_pipe.elements[-1][0]))
                else:
                    self.results.best_config_feature_importances = feature_importances

                    # write backmapping file only if optimum_pipes inverse_transform works completely.
                    # restriction: only a faulty inverse_transform is considered, missing ones are further ignored.
                    with warnings.catch_warnings(record=True) as w:
                        # get backmapping
                        backmapping, _, _ = self.optimum_pipe.\
                            inverse_transform(np.array(feature_importances).reshape(1, -1), None)

                        if not any("The inverse transformation is not possible for" in s
                                   for s in [e.message.args[0] for e in w]):
                            # save backmapping
                            self.results_handler.save_backmapping(
                                filename='optimum_pipe_feature_importances_backmapped', backmapping=backmapping)
                        else:
                            logger.info('Could not save feature importance: backmapping NOT successful.')

                # save learning curves
                if self.cross_validation.learning_curves:
                    self.results_handler.save_all_learning_curves()

        logger.info("Summarizing results...")

        logger.info("Write predictions to files...")
        # write all convenience files (summary, predictions_file and plots)
        self.results_handler.write_predictions_file()

        logger.info("Write summary...")
        logger.stars()
        logger.photon_system_log("")
        logger.photon_system_log(self.results_handler.text_summary())

    def preprocess_data(self):
        # if there is a preprocessing pipeline, we apply it first.
        if self.preprocessing is not None:
            logger.info("Applying preprocessing steps...")
            self.preprocessing.fit(self.data.X, self.data.y, **self.data.kwargs)
            self.data.X, self.data.y, self.data.kwargs = self.preprocessing.transform(self.data.X, self.data.y,
                                                                                      **self.data.kwargs)

    def _prepare_pipeline(self):
        self._pipe = Branch.prepare_photon_pipe(self.elements)
        self._pipe = Branch.sanity_check_pipeline(self._pipe)
        if self.random_state:
            self._pipe.random_state = self.random_state

    # ===================================================================
    # sklearn interfaces
    # ===================================================================

    @staticmethod
    def fit_outer_folds(outer_fold_computer, X, y, kwargs, cache_folder):
        try:
            outer_fold_computer.fit(X, y, **kwargs)
        finally:
            CacheManager.clear_cache_files(cache_folder)
        return

    def fit(self, data: np.ndarray, targets: np.ndarray, **kwargs):
        """
        Starts the hyperparameter search and/or fits the pipeline to the data and targets.

        Manages the nested cross validated hyperparameter search:

        1. Filters the data according to filter strategy (1) and according to the imbalanced_data_strategy (2)
        2. requests new configurations from the hyperparameter search strategy, the optimizer,
        3. initializes the testing of a specific configuration,
        4. communicates the result to the optimizer,
        5. repeats 2-4 until optimizer delivers no more configurations to test
        6. finally searches for the best config in all tested configs,
        7. trains the pipeline with the best config and evaluates the performance on the test set

        Parameters:
            data:
                The array-like training and test data with shape=[N, D],
                where N is the number of samples and D is the number of features.

            targets:
                The truth array-like values with shape=[N],
                where N is the number of samples.

            **kwargs:
                Keyword arguments, passed to Outer_Fold_Manager.fit.


        Returns:
            Fitted Hyperpipe.

        """
        # switch to result output folder
        start = datetime.datetime.now()
        self.output_settings.update_settings(self.name, start.strftime("%Y-%m-%d_%H-%M-%S"))

        logger.photon_system_log('=' * 101)
        logger.photon_system_log('PHOTONAI ANALYSIS: ' + self.name)
        logger.photon_system_log('=' * 101)
        logger.info("Preparing data and PHOTONAI objects for analysis...")

        # loop over outer cross validation
        if self.nr_of_processes > 1:
            hyperpipe_client = Client(threads_per_worker=1, n_workers=self.nr_of_processes, processes=False)

        try:
            # check data
            self.data.input_data_sanity_checks(data, targets, **kwargs)
            # create photon pipeline
            self._prepare_pipeline()
            # initialize the progress monitors
            self._prepare_result_logging(start)
            # apply preprocessing
            self.preprocess_data()

            if not self.is_final_fit:

                # Outer Folds
                outer_folds = FoldInfo.generate_folds(self.cross_validation.outer_cv,
                                                      self.data.X, self.data.y, self.data.kwargs,
                                                      self.cross_validation.use_test_set,
                                                      self.cross_validation.test_size)

                self.cross_validation.outer_folds = {f.fold_id: f for f in outer_folds}
                delayed_jobs = []

                # Run Dummy Estimator
                dummy_estimator = self._prepare_dummy_estimator()

                if self.cache_folder is not None:
                    logger.info("Removing cache files...")
                    CacheManager.clear_cache_files(self.cache_folder, force_all=True)

                # loop over outer cross validation
                for i, outer_f in enumerate(outer_folds):

                    # 1. generate OuterFolds Object
                    outer_fold = MDBOuterFold(fold_nr=outer_f.fold_nr)
                    outer_fold_computer = OuterFoldManager(self._pipe,
                                                           self.optimization,
                                                           outer_f.fold_id,
                                                           self.cross_validation,
                                                           cache_folder=self.cache_folder,
                                                           cache_updater=self.recursive_cache_folder_propagation,
                                                           dummy_estimator=dummy_estimator,
                                                           result_obj=outer_fold)
                    # 2. monitor outputs
                    self.results.outer_folds.append(outer_fold)

                    if self.nr_of_processes > 1:
                        result = dask.delayed(Hyperpipe.fit_outer_folds)(outer_fold_computer,
                                                                         self.data.X,
                                                                         self.data.y,
                                                                         self.data.kwargs,
                                                                         self.cache_folder)
                        delayed_jobs.append(result)
                    else:
                        try:
                            # 3. fit
                            outer_fold_computer.fit(self.data.X, self.data.y, **self.data.kwargs)
                            # 4. save outer fold results
                            self.results_handler.save()
                        finally:
                            # 5. clear cache
                            CacheManager.clear_cache_files(self.cache_folder)

                if self.nr_of_processes > 1:
                    dask.compute(*delayed_jobs)
                    self.results_handler.save()

                # evaluate hyperparameter optimization results for best config
                self._finalize_optimization()

                # clear complete cache ?
                CacheManager.clear_cache_files(self.cache_folder, force_all=True)

            ###############################################################################################
            else:
                self.preprocess_data()
                self._pipe.fit(self.data.X, self.data.y, **kwargs)
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            traceback.print_exc()
            raise e
        finally:
            if self.nr_of_processes > 1:
                hyperpipe_client.close()
        return self

    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the optimum pipe to predict the input data.

        Parameters:
            data:
                The array-like prediction data with shape=[M, D],
                where M is the number of samples and D is the number
                of features. D must correspond to the number
                of trained dimensions of the fit method.

            **kwargs:
                Keyword arguments, passed to optimum_pipe.predict.

        Returns:
            Predicted targets calculated on input data with trained model.

        """
        # Todo: if local_search = true then use optimized pipe here?
        if self._pipe:
            return self.optimum_pipe.predict(data, **kwargs)

    def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the optimum pipe to predict the probabilities from the input data.

        Parameters:
            data:
                The array-like prediction data with shape=[M, D],
                where M is the number of samples and D is the number
                of features. D must correspond to the number
                of trained dimensions of the fit method.

            **kwargs:
                Keyword arguments, passed to optimum_pipe.predict_proba.

        Returns:
            Probabilities calculated from input data on fitted model.


        """
        if self._pipe:
            return self.optimum_pipe.predict_proba(data, **kwargs)

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the optimum pipe to transform the data.

        Parameters:
            data:
                The array-like input data with shape=[M, D],
                where M is the number of samples and D is the number
                of features. D must correspond to the number
                of trained dimensions of the fit method.

            **kwargs:
                Keyword arguments, passed to optimum_pipe.transform.

        Returns:
            Transformed data.

        """
        if self._pipe:
            X, _, _ = self.optimum_pipe.transform(data, y=None, **kwargs)
            return X

    def score(self, data: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Use the optimum pipe to score the model.

        Parameters:
            data:
                The array-like data with shape=[M, D],
                where M is the number of samples and D is the number
                of features. D must correspond to the number
                of trained dimensions of the fit method.

            y:
                The array-like true targets.

            **kwargs:
                Keyword arguments, passed to optimum_pipe.predict.

        Returns:
            Score data on input data with trained model.

        """
        if self._pipe:
            predictions = self.optimum_pipe.predict(data, **kwargs)
            scorer = Scorer.create(self.optimization.best_config_metric)
            return scorer(y, predictions)

    def get_permutation_feature_importances(self, X_val: np.ndarray, y_val: np.ndarray, **kwargs):
        """
        Since PHOTONAI is built on top of the scikit-learn interface,
        it is possible to use direct functions from their package.
        Here the example of the [feature importance via permutations](
        https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).

        Parameters:
            X_val:
                The array-like data with shape=[M, D],
                where M is the number of samples and D is the number
                of features. D must correspond to the number
                of trained dimensions of the fit method.

            y_val:
                The array-like true targets.

            **kwargs:
                Keyword arguments, passed to sklearn.permutation_importance.

        Returns:
            Dictionary-like object, with the following attributes: importances_mean, importances_std, importances.

        """

        return permutation_importance(self.optimum_pipe, X_val, y_val, **kwargs)

    def inverse_transform_pipeline(self, hyperparameters: dict,
                                   data: np.ndarray,
                                   targets: np.ndarray,
                                   data_to_inverse: np.ndarray) -> np.ndarray:
        """
        Inverse transform data for a pipeline with specific hyperparameter configuration.

        1. Copy Sklearn Pipeline,
        2. Set Parameters
        3. Fit Pipeline to data and targets
        4. Inverse transform data with that pipeline

        Parameters:
            hyperparameters:
                The concrete configuration settings for the pipeline elements.

            data:
                The training data to which the pipeline is fitted.

            targets:
                The truth values for training.

            data_to_inverse:
                The data that should be inversed after training.

        Returns:
            Inverse data as array.

        """
        copied_pipe = self.pipe.copy_me()
        copied_pipe.set_params(**hyperparameters)
        copied_pipe.fit(data, targets)
        return copied_pipe.inverse_transform(data_to_inverse)

    # ===================================================================
    # Copy, Save and Load
    # ===================================================================

    def copy_me(self):
        """
        Helper function to copy an entire Hyperpipe

        Returns:
            Hyperpipe

        """
        signature = inspect.getfullargspec(OutputSettings.__init__)[0]
        settings = OutputSettings()
        for attr in signature:
            if hasattr(self.output_settings, attr):
                setattr(settings, attr, getattr(self.output_settings, attr))
        self.output_settings.initialize_log_file()

        # create new Hyperpipe instance
        pipe_copy = Hyperpipe(name=self.name,
                              inner_cv=deepcopy(self.cross_validation.inner_cv),
                              outer_cv=deepcopy(self.cross_validation.outer_cv),
                              best_config_metric=self.optimization.best_config_metric,
                              metrics=self.optimization.metrics,
                              optimizer=self.optimization.optimizer_input_str,
                              optimizer_params=self.optimization.optimizer_params,
                              project_folder=self.project_folder,
                              output_settings=settings)

        signature = inspect.getfullargspec(self.__init__)[0]
        for attr in signature:
            if hasattr(self, attr) and attr != 'output_settings':
                setattr(pipe_copy, attr, getattr(self, attr))

        if hasattr(self, 'preprocessing') and self.preprocessing:
            preprocessing = Preprocessing()
            for element in self.preprocessing.elements:
                preprocessing += element.copy_me()
            pipe_copy += preprocessing
        if hasattr(self, 'elements'):
            for element in self.elements:
                pipe_copy += element.copy_me()
        return pipe_copy

    def save_optimum_pipe(self, filename=None, password=None):
        if filename is None:
            filename = "photon_" + self.name + "_best_model.p"
        PhotonModelPersistor.save_optimum_pipe(self, filename, password)

    @staticmethod
    def load_optimum_pipe(file: str, password: str = None) -> PhotonPipeline:
        """
        Load optimum pipe from file.
        As staticmethod, instantiation is thus not required.
        Called backend: PhotonModelPersistor.load_optimum_pipe.

        Parameters:
            file:
                File path specifying .photon file to load
                trained pipeline from zipped file.

            password:
                Passcode for read file.

        Returns:
            Returns pipeline with all trained PipelineElements.

        """
        return PhotonModelPersistor.load_optimum_pipe(file, password)

    def __repr__(self, **kwargs):
        """Overwrite BaseEstimator's function to avoid errors when using Jupyter Notebooks."""
        return "Hyperpipe(name='{}')".format(self.name)


class PhotonModelPersistor:

    @staticmethod
    def save_elements(elements, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        element_identifier = list()

        for i, element in enumerate(elements):
            if hasattr(element, 'disabled'):
                if element.disabled:
                    continue

            # Switch in Switch not covered!
            if isinstance(element, Switch):
                element = element.base_element

            if isinstance(element, (Stack, Branch, Preprocessing)):
                filename = '_' + str(i) + '_' + element.name
                new_folder = os.path.join(folder, filename)
                element_identifier.append({'element_name': element.name, 'filename': filename})
                elements = element.elements
                PhotonModelPersistor.save_elements(elements=elements, folder=new_folder)
                element.elements = []
                joblib.dump(element, os.path.join(folder, filename) + '.pkl', compress=1)
                element_identifier[-1]['mode'] = 'PhotonBuildingBlock'
                element.elements = elements
            else:
                if not hasattr(element, 'base_element'):
                    base_element = element
                else:
                    base_element = element.base_element
                filename = '_' + str(i) + '_' + element.name
                element_identifier.append({'element_name': element.name, 'filename': filename})
                if hasattr(base_element, 'save'):
                    wrapper_file = inspect.getfile(base_element.__class__)
                    base_element.save(os.path.join(folder, filename))
                    element_identifier[-1]['mode'] = 'custom'
                    element_identifier[-1]['wrapper_script'] = os.path.basename(wrapper_file)
                    element_identifier[-1]['test_disabled'] = element.test_disabled
                    element_identifier[-1]['disabled'] = element.disabled
                    element_identifier[-1]['hyperparameters'] = element.hyperparameters
                    # (class_name != element_name) - possibility
                    element_identifier[-1]['class_name'] = type(base_element).__name__
                    shutil.copy(wrapper_file, os.path.join(folder, os.path.basename(wrapper_file)))
                else:
                    try:
                        joblib.dump(element, os.path.join(folder, filename) + '.pkl', compress=1)
                        element_identifier[-1]['mode'] = 'pickle'
                    except:
                        raise NotImplementedError("Custom pipeline element must implement .save() method or "
                                                  "allow pickle.")

        # save pipeline blueprint to make loading of pipeline easier
        with open(os.path.join(folder, '_optimum_pipe_blueprint.pkl'), 'wb') as f:
            pickle.dump(element_identifier, f)

    @staticmethod
    def save_optimum_pipe(optimum_pipe, zip_file: str, password: str = None):
        """
        Save optimal pipeline only. Complete hyperpipe will no not be saved.

        Parameters:
            optimum_pipe:
                The optimum pipe to save.

            zip_file:
                File path as string specifying file to save pipeline to.

            password:
                Password used to encrypt the pipeline file.

        """
        folder = os.path.splitext(zip_file)[0]
        zip_file = folder + '.photon'

        if os.path.exists(folder):
            msg = 'The file you specified already exists as a folder.'
            logger.warning(msg)
            warnings.warn(msg)
        else:
            os.makedirs(folder)

        # only save elements without name. Structure of optimum_pipe.elements: [('name', element),...]
        PhotonModelPersistor.save_elements([val[1] for val in optimum_pipe.elements], folder)

        # write meta infos from pipeline
        with open(os.path.join(folder, '_optimum_pipe_meta.pkl'), 'wb') as f:
            meta_infos = {'photon_version': __version__}
            pickle.dump(meta_infos, f)

        # get all files
        files = list()
        for root, directories, filenames in os.walk(folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        if password is not None:
            import pyminizip
            pyminizip.compress(files, zip_file, password)
        else:
            with zipfile.ZipFile(zip_file, 'w') as myzip:
                root_len = len(os.path.dirname(zip_file)) + 1
                for f in files:
                    # in order to work even with subdirectories, we need to substract the dirname from our file
                    # this is why I'm saving the root_len first
                    myzip.write(f, f[root_len:])
                    os.remove(f)
        shutil.rmtree(folder)

    @staticmethod
    def load_elements(folder):
        with open(os.path.join(folder, '_optimum_pipe_blueprint.pkl'), 'rb') as f:
            setup_info = pickle.load(f)
            element_list = list()
            for element_info in setup_info:
                if element_info['mode'] == 'PhotonBuildingBlock':
                    photon_building_block = joblib.load(os.path.join(folder, element_info['filename'] + '.pkl'))
                    base_elements = PhotonModelPersistor.load_elements(os.path.join(folder, element_info['filename']))
                    for _, element in base_elements:
                        photon_building_block += element
                    element_list.append((element_info['element_name'], photon_building_block))
                elif element_info['mode'] == 'custom':
                    if 'class_name' in element_info:  # (class_name != element_name) - possibility
                        spec = importlib.util.spec_from_file_location(element_info['class_name'],
                                                                      os.path.join(folder,
                                                                                   element_info['wrapper_script']))
                        imported_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(imported_module)
                        base_element = getattr(imported_module, element_info['class_name'])
                    else:
                        # backward compatibility
                        try:
                            spec = importlib.util.spec_from_file_location(element_info['element_name'],
                                                                          os.path.join(folder,
                                                                          element_info['wrapper_script']))
                            imported_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(imported_module)
                            base_element = getattr(imported_module, element_info['element_name'])
                        except:
                            msg = "Outdated version: The imported module was created by an older photon version. " \
                                  "Please retrain your model with a newer version."
                            logger.error(msg)
                            raise RuntimeError(msg)
                    custom_element = PipelineElement(name=element_info['element_name'], base_element=base_element(),
                                                     hyperparameters=element_info['hyperparameters'],
                                                     test_disabled=element_info['test_disabled'],
                                                     disabled=element_info['disabled'])
                    custom_element.base_element.load(os.path.join(folder, element_info['filename']))
                    element_list.append((element_info['element_name'], custom_element))
                else:
                    loaded_pipeline_element = joblib.load(os.path.join(folder, element_info['filename'] + '.pkl'))
                    element_list.append((element_info['element_name'], loaded_pipeline_element))
        return element_list

    @staticmethod
    def load_optimum_pipe(file: str, password: str = None):
        """
        Load optimum pipe from file.
        As staticmethod, instantiation is thus not required.
        Called backend: PhotonModelPersistor.load_optimum_pipe.

        Parameters:
            file:
                File path specifying .photon file to load
                trained pipeline from zipped file.

            password:
                Passcode for read file.

        Returns:
            Returns pipeline with all trained PipelineElements.

        """
        if file.endswith('.photon'):
            folder = os.path.dirname(file)
            zf = zipfile.ZipFile(file)
            zf.extractall(folder, pwd=password)
        else:
            raise FileNotFoundError('Specify .photon file that holds PHOTON optimum pipe.')

        load_folder = os.path.join(folder, 'photon_best_model')
        meta_infos = {}
        try:
            with open(os.path.join(load_folder, '_optimum_pipe_meta.pkl'), 'rb') as f:
                meta_infos = pickle.load(f)
        except:
            print("Could not load meta information for optimum pipe")

        element_list = PhotonModelPersistor.load_elements(folder=load_folder)

        # delete unpacked folder to clean up
        # ToDo: Don't unpack at all, but use PHOTON file directly
        from shutil import rmtree
        rmtree(os.path.join(folder, 'photon_best_model'), ignore_errors=True)

        photon_pipe = PhotonPipeline(element_list)
        photon_pipe._meta_information = meta_infos
        return photon_pipe
