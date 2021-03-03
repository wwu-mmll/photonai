import json
import time
import traceback
import warnings
import datetime
import numpy as np
from typing import Union, List

from photonai.helper.helper import PhotonPrintHelper, PhotonDataHelper, print_double_metrics
from photonai.optimization.performance_constraints import PhotonBaseConstraint
from photonai.photonlogger.logger import logger
from photonai.processing.metrics import Scorer
from photonai.processing.results_structure import MDBHelper, MDBInnerFold, MDBScoreInformation, MDBFoldMetric, MDBConfig

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class InnerFoldManager(object):
    """Inner Fold manager.

    Manages the inner part of the cross validation.

    Information about the inner folds is managed in this object.
    It operates under a specific hyperparameter configuration,
    triggers unified learning, and provides metrics about all inner folds.

    Parameters
    ----------
    pipe_ctor
        The pipeline instance that shall be trained and tested.

    specific_config: dict
        The hyperparameter configuration to test

    optimization_infos: Optimization
        Informations for the optimizer like best_config_metric, maximize_metric, optimizer_params, ...

    cross_validation_infos: CrossValidation
        Infomrations for the inner cross-validation like test_size, eval_final_performance, ...

    outer_fold_id: UUID
        UUID for outer_fold for identification.

    optimization_constraints: PhotonBaseConstraint or List of PhotonBaseConstraint
        Constraints for skipping folds of config if specific constraint occurs.

    raise_error: bool, default=False
        if true, raises exception when training and testing the pipeline fails

    training: bool, default=False
        Mode switch.

    cache_folder: str, default=None
        Path to cache in.

    cache_updater: default=None
        Funtion to update cache.

    """

    def __init__(self, pipe_ctor, specific_config: dict, optimization_infos,
                 cross_validation_infos, outer_fold_id,
                 optimization_constraints: Union[PhotonBaseConstraint, List[PhotonBaseConstraint]] = None,
                 raise_error: bool = False,
                 training: bool = False,
                 cache_folder=None,
                 cache_updater=None):

        self.params = specific_config
        self.pipe = pipe_ctor
        self.optimization_infos = optimization_infos
        self.optimization_constraints = optimization_constraints
        self.outer_fold_id = outer_fold_id
        self.cross_validation_infos = cross_validation_infos

        self.cache_folder = cache_folder
        self.cache_updater = cache_updater

        self.raise_error = raise_error
        self.training = training

    def fit(self, X, y, **kwargs):
        """Iterates over cross-validation folds and trains the pipeline,
        then uses it for predictions. Calculates metrics per fold and averages
        them over fold.

        Parameters
        ----------
        X: ndarray
            Training and test data

        y: ndarray
            Training and test targets

        Returns
        -------
        config_item: MDBConfig
            configuration class for result tree that monitors training and test performance

        """

        config_item = MDBConfig()
        config_item.config_dict = self.params
        config_item.inner_folds = []
        config_item.metrics_test = []
        config_item.metrics_train = []
        config_item.computation_start_time = datetime.datetime.now()

        try:
            # do inner cv
            for idx, (inner_fold_id, inner_fold) in enumerate(self.cross_validation_infos.inner_folds[self.outer_fold_id].items()):

                train, test = inner_fold.train_indices, inner_fold.test_indices

                # split kwargs according to cross validation
                train_X, train_y, kwargs_cv_train = PhotonDataHelper.split_data(X, y, kwargs, indices=train)
                test_X, test_y, kwargs_cv_test = PhotonDataHelper.split_data(X, y, kwargs, indices=test)

                new_pipe = self.pipe()
                if self.cache_folder is not None and self.cache_updater is not None:
                    self.cache_updater(new_pipe, self.cache_folder, inner_fold_id)

                if not config_item.human_readable_config:
                    config_item.human_readable_config = PhotonPrintHelper.config_to_human_readable_dict(new_pipe,
                                                                                                        self.params)
                    logger.clean_info(json.dumps(config_item.human_readable_config, indent=4, sort_keys=True))

                job_data = InnerFoldManager.InnerCVJob(pipe=new_pipe,
                                                       config=dict(self.params),
                                                       metrics=self.optimization_infos.metrics,
                                                       callbacks=self.optimization_constraints,
                                                       train_data=InnerFoldManager.JobData(train_X, train_y, train,
                                                                                           kwargs_cv_train),
                                                       test_data=InnerFoldManager.JobData(test_X, test_y, test,
                                                                                          kwargs_cv_test))

                # only for unparallel processing
                # inform children in which inner fold we are
                # self.pipe.distribute_cv_info_to_hyperpipe_children(inner_fold_counter=fold_cnt)
                # self.mother_inner_fold_handle(fold_cnt)

                # --> write that output in InnerFoldManager!
                # logger.debug(config_item.human_readable_config)
                fold_nr = idx + 1
                logger.debug('calculating inner fold ' + str(fold_nr) + '...')

                curr_test_fold, curr_train_fold = InnerFoldManager.fit_and_score(job_data)
                if self.cross_validation_infos.learning_curves:
                    learning_curves = self.compute_learning_curves(new_pipe, train_X, train_y, train, kwargs_cv_train,
                                                                   test_X, test_y, test, kwargs_cv_test)
                    learning_curves.append([1., curr_test_fold.metrics, curr_train_fold.metrics])
                else:
                    learning_curves = list()

                logger.debug('Performance inner fold ' + str(fold_nr))
                print_double_metrics(curr_train_fold.metrics, curr_test_fold.metrics, photon_system_log=False)

                durations = job_data.pipe.time_monitor

                self.update_config_item_with_inner_fold(config_item=config_item,
                                                        fold_cnt=fold_nr,
                                                        curr_train_fold=curr_train_fold,
                                                        curr_test_fold=curr_test_fold,
                                                        time_monitor=durations,
                                                        feature_importances=new_pipe.feature_importances_,
                                                        learning_curves=learning_curves)

                if isinstance(self.optimization_constraints, list):
                    break_cv = 0
                    for cf in self.optimization_constraints:
                        if not cf.shall_continue(config_item):
                            logger.info('Skipped further cross validation after fold ' + str(fold_nr) +
                                        ' due to performance constraints in ' + cf.metric)
                            break_cv += 1
                            break
                    if break_cv > 0:
                        break
                elif isinstance(self.optimization_constraints, PhotonBaseConstraint):
                    if not self.optimization_constraints.shall_continue(config_item):
                        logger.info('Skipped further cross validation after fold ' + str(fold_nr) +
                                    ' due to performance constraints in ' + self.optimization_constraints.metric)
                        break

            InnerFoldManager.process_fit_results(config_item,
                                                 self.cross_validation_infos.calculate_metrics_across_folds,
                                                 self.cross_validation_infos.calculate_metrics_per_fold,
                                                 self.optimization_infos.metrics)

        except Exception as e:
            if self.raise_error:
                raise e
            logger.error(e)
            logger.error(traceback.format_exc())
            traceback.print_exc()
            if not isinstance(e, Warning):
                config_item.config_failed = True
            config_item.config_error = str(e)
            warnings.warn('One test iteration of pipeline failed with error')

        logger.debug('...done with')
        logger.debug(json.dumps(config_item.human_readable_config, indent=4, sort_keys=True))

        config_item.computation_end_time = datetime.datetime.now()
        return config_item

    def compute_learning_curves(self, new_pipe, train_X, train_y, train, kwargs_cv_train,
                                test_X, test_y, test, kwargs_cv_test):
        self.cross_validation_infos.learning_curves_cut.transform()
        cut_range = [round(cut * train_X.shape[0]) for cut in self.cross_validation_infos.learning_curves_cut.values]
        learning_curves = []
        for i, cut in enumerate(cut_range[1:]):
            cut_indices = np.arange(cut)
            train_cut_X, train_cut_y, train_cut_kwargs = PhotonDataHelper.split_data(train_X, train_y, kwargs_cv_train,
                                                                                     indices=cut_indices)
            train_cut = train[:cut]
            job_data = self.InnerCVJob(pipe=new_pipe,
                                       config=dict(self.params),
                                       metrics=self.optimization_infos.metrics,
                                       callbacks=self.optimization_constraints,
                                       train_data=self.JobData(train_cut_X, train_cut_y, train_cut, train_cut_kwargs),
                                       test_data=self.JobData(test_X, test_y, test, kwargs_cv_test))
            curr_test_cut, curr_train_cut = InnerFoldManager.fit_and_score(job_data)
            learning_curves.append([self.cross_validation_infos.learning_curves_cut.values[i], curr_test_cut.metrics,
                                    curr_train_cut.metrics])
        return learning_curves

    class JobData:
        def __init__(self, X, y, indices, cv_kwargs):
            self.X = X
            self.y = y
            self.indices = indices
            self.cv_kwargs = cv_kwargs

    class InnerCVJob:

        def __init__(self, pipe, config, metrics, callbacks, train_data, test_data):
            self.pipe = pipe
            self.config = config
            self.metrics = metrics
            self.callbacks = callbacks
            self.train_data = train_data
            self.test_data = test_data

    @staticmethod
    def update_config_item_with_inner_fold(config_item, fold_cnt, curr_train_fold, curr_test_fold, time_monitor,
                                           feature_importances, learning_curves):
        # fill result tree with fold information
        inner_fold = MDBInnerFold()
        inner_fold.fold_nr = fold_cnt
        inner_fold.training = curr_train_fold
        inner_fold.validation = curr_test_fold

        inner_fold.number_samples_validation = len(curr_test_fold.indices)
        inner_fold.number_samples_training = len(curr_train_fold.indices)
        inner_fold.time_monitor = time_monitor
        inner_fold.feature_importances = feature_importances
        inner_fold.learning_curves = learning_curves
        # save all inner folds to the tree under the config item
        config_item.inner_folds.append(inner_fold)

    @staticmethod
    def process_fit_results(config_item,
                            calculate_metrics_across_folds,
                            calculate_metrics_per_fold,
                            metrics):

        overall_y_pred_test = []
        overall_y_true_test = []
        overall_y_pred_train = []
        overall_y_true_train = []

        for fold in config_item.inner_folds:
            curr_test_fold = fold.validation
            curr_train_fold = fold.training

            if calculate_metrics_across_folds:
                # if we have one hot encoded values -> concat horizontally
                if isinstance(curr_test_fold.y_pred, np.ndarray):
                    if len(curr_test_fold.y_pred.shape) > 1:
                        axis = 1
                    else:
                        axis = 0
                else:
                    # if we have lists concat
                    axis = 0
                overall_y_true_test = np.concatenate((overall_y_true_test, curr_test_fold.y_true), axis=axis)
                overall_y_pred_test = np.concatenate((overall_y_pred_test, curr_test_fold.y_pred), axis=axis)

                # we assume y_pred from the training set comes in the same shape as y_pred from the test se
                overall_y_true_train = np.concatenate((overall_y_true_train, curr_train_fold.y_true), axis=axis)
                overall_y_pred_train = np.concatenate((overall_y_pred_train, curr_train_fold.y_pred), axis=axis)

                # metrics across folds
                metrics_to_calculate = list(metrics)
                if 'score' in metrics_to_calculate:
                    metrics_to_calculate.remove('score')
                metrics_train = Scorer.calculate_metrics(overall_y_true_train,
                                                         overall_y_pred_train, metrics_to_calculate)
                metrics_test = Scorer.calculate_metrics(overall_y_true_test,
                                                        overall_y_pred_test, metrics_to_calculate)

                def metric_to_db_class(metric_list):
                    db_metrics = []
                    for metric_name, metric_value in metric_list.items():
                        new_metric = MDBFoldMetric(operation="raw", metric_name=metric_name,
                                                   value=metric_value)
                        db_metrics.append(new_metric)
                    return db_metrics

                db_metrics_train = metric_to_db_class(metrics_train)
                db_metrics_test = metric_to_db_class(metrics_test)

                # if we want to have metrics for each fold as well, calculate mean and std.
                if calculate_metrics_per_fold:
                    db_metrics_fold_train, db_metrics_fold_test = MDBHelper.aggregate_metrics_for_inner_folds(config_item.inner_folds,
                                                                                                              metrics)
                    config_item.metrics_train = db_metrics_train + db_metrics_fold_train
                    config_item.metrics_test = db_metrics_test + db_metrics_fold_test
                else:
                    config_item.metrics_train = db_metrics_train
                    config_item.metrics_test = db_metrics_test

            elif calculate_metrics_per_fold:
                # calculate mean and std over all fold metrics
                config_item.metrics_train, config_item.metrics_test = MDBHelper.aggregate_metrics_for_inner_folds(config_item.inner_folds,
                                                                                                                  metrics)

    @staticmethod
    def fit_and_score(job: InnerCVJob):

        pipe = job.pipe

        # set params to current config
        pipe.set_params(**job.config)

        # start fitting
        pipe.fit(job.train_data.X, job.train_data.y, **job.train_data.cv_kwargs)

        logger.debug('Scoring Training Data')

        # score test data
        curr_test_fold = InnerFoldManager.score(pipe, job.test_data.X, job.test_data.y, job.metrics,
                                                indices=job.test_data.indices,
                                                **job.test_data.cv_kwargs)

        logger.debug('Scoring Test Data')
        # score train data
        curr_train_fold = InnerFoldManager.score(pipe, job.train_data.X, job.train_data.y, job.metrics,
                                                 indices=job.train_data.indices,
                                                 training=True, **job.train_data.cv_kwargs)

        return curr_test_fold, curr_train_fold

    @staticmethod
    def score(estimator, X, y_true, metrics, indices=[],
              calculate_metrics: bool=True, training: bool=False, **kwargs):
        """Uses the pipeline to predict the given data,
        compare it to the truth values and calculate metrics

        Parameters
        ----------
        estimator:
            The pipeline or pipeline element for prediction.
            Requires the 'predict' method.


        X: ndarray
            The data for prediction.

        y_true: ndarray
            The truth values for the data.

        metrics: list
            The metrics to be calculated.

        indices: list
            The indices of the given data and targets that are logged into the result tree.

        training: list
            if True, all training_only pipeline elements are executed, if False they are skipped

        calculate_metrics: bool, default=True
            if True, calculates metrics for given data

        training: bool, default=False
            If True, an estimator.transform() is prepended here.


        Returns
        -------
        information: MDBScoreInformation
            Informations about scorer.

        """

        scoring_time_start = time.time()

        output_metrics = {}

        if not training:
            y_pred = estimator.predict(X, **kwargs)
        else:
            X, y_true_new, kwargs_new = estimator.transform(X, y_true, **kwargs)
            if y_true_new is not None:
                y_true = y_true_new
            if kwargs_new is not None and len(kwargs_new) > 0:
                kwargs = kwargs_new
            y_pred = estimator.predict(X, training=True, **kwargs)

        # Nice to have
        # InnerFoldManager.plot_some_data(y_true, y_pred)

        if calculate_metrics:
            if isinstance(y_pred, np.ndarray) and y_pred.dtype.names:
                y_pred_names = [y_pred.dtype.names]
                if "y_pred" not in y_pred_names[0]:
                    msg = "If scorer object does not return 1d array or list, PHOTON expected name 'y_pred' in nd array."
                    logger.error(msg)
                    raise KeyError(msg)
                score_metrics = Scorer.calculate_metrics(y_true, y_pred["y_pred"], metrics)
            else:
                y_pred_names = []
                score_metrics = Scorer.calculate_metrics(y_true, y_pred, metrics)

            # add default metric
            if output_metrics:
                output_metrics = {**output_metrics, **score_metrics}
            else:
                output_metrics = score_metrics
        else:
            output_metrics = {}

        final_scoring_time = time.time() - scoring_time_start

        probabilities = []
        if hasattr(estimator, '_final_estimator'):
            if hasattr(estimator._final_estimator.base_element, 'predict_proba'):
                probabilities = estimator.predict_proba(X, training=training,  **kwargs)

                try:
                    if probabilities is not None:
                        if not len(probabilities) == 0:
                            probabilities = probabilities.tolist()
                except:
                    warnings.warn('No probabilities available.')

        if not isinstance(y_pred, list):
            y_pred = y_pred_names + np.asarray(y_pred).tolist()

        if not isinstance(y_true, list):
            y_true = np.asarray(y_true).tolist()

        score_result_object = MDBScoreInformation(metrics=output_metrics,
                                                  score_duration=final_scoring_time,
                                                  y_pred=y_pred, y_true=y_true,
                                                  indices=np.asarray(indices).tolist(),
                                                  probabilities=probabilities)

        return score_result_object
