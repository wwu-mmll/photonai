import time
import traceback
import warnings
import os

import numpy as np
from sklearn.pipeline import Pipeline

from multiprocessing import Process, Queue
import queue

from ..photonlogger.Logger import Logger
from ..validation.ResultsDatabase import MDBHelper, MDBInnerFold, MDBScoreInformation, MDBFoldMetric, \
    FoldOperations, MDBConfig


class TestPipeline(object):
    """
        Trains and tests a sklearn pipeline for a specific hyperparameter combination with cross-validation,
        calculates metrics for each fold and averages metrics over all folds
    """

    def __init__(self, pipe_ctor, specific_config: dict, optimization_infos,
                 cross_validation_infos, outer_fold_id,
                 raise_error: bool=False, save_predictions: bool=False, save_feature_importances: bool=False,
                 training: bool = False, cache_folder = None, cache_updater = None,
                 parallel_cv: bool = False, nr_of_parrallel_processes: int = 4):
        """
        Creates a new TestPipeline object
        :param pipe: The sklearn pipeline instance that shall be trained and tested
        :type pipe: Pipeline
        :param specific_config: The hyperparameter configuration to test
        :type specific_config: dict
        :param raise_error: if true, raises exception when training and testing the pipeline fails
        :type raise_error: bool
        """

        self.params = specific_config
        self.pipe = pipe_ctor
        self.optimization_infos = optimization_infos
        self.outer_fold_id = outer_fold_id
        self.cross_validation_infos = cross_validation_infos

        self.save_predictions = save_predictions
        self.save_feature_importances = save_feature_importances
        self.cache_folder = cache_folder
        self.cache_updater = cache_updater

        self.raise_error = raise_error
        self.training = training

        self.parallel_cv = parallel_cv
        self.nr_of_parallel_processes = nr_of_parrallel_processes

    def fit(self, X, y, **kwargs):
        """
        Iterates over cross-validation folds and trains the pipeline, then uses it for predictions.
        Calculates metrics per fold and averages them over fold.
        :param X: Training and test data
        :param y: Training and test targets
        :returns: configuration class for result tree that monitors training and test performance
        """

        # needed for testing Timeboxed Random Grid Search
        # time.sleep(35)

        config_item = MDBConfig()
        config_item.inner_folds = []
        config_item.metrics_test = []
        config_item.metrics_train = []
        fold_cnt = 0

        # if we want to collect the predictions, we need to save them into the tree
        original_save_predictions = self.save_predictions
        save_predictions = bool(self.save_predictions)
        save_feature_importances = self.save_feature_importances

        if self.cross_validation_infos.calculate_metrics_across_folds:
            save_predictions = True

        if self.parallel_cv:
            nr_of_processes = min(self.nr_of_parallel_processes, )
            folds_to_do = Queue()
            folds_done = Queue()

        list_of_score_results = []

        try:
            # do inner cv
            for inner_fold_id, inner_fold in self.cross_validation_infos.inner_folds[self.outer_fold_id].items():

                train, test = inner_fold.train_indices, inner_fold.test_indices

                # split kwargs according to cross validation
                kwargs_cv_train = {}
                kwargs_cv_test = {}
                if len(kwargs) > 0:
                    for name, sublist in kwargs.items():
                        if isinstance(sublist, (list, np.ndarray)):
                            kwargs_cv_train[name] = sublist[train]
                            kwargs_cv_test[name] = sublist[test]

                new_pipe = self.pipe()
                if self.cache_folder is not None and self.cache_updater is not None:
                    self.cache_updater(new_pipe, self.cache_folder, inner_fold_id)

                job_data = TestPipeline.InnerCVJob(pipe=new_pipe,
                                                   config=dict(self.params),
                                                   metrics=list(self.optimization_infos.metrics),
                                                   callbacks=self.optimization_infos.inner_cv_callback_functions,
                                                   train_data=TestPipeline.JobData(X[train], y[train], train, dict(kwargs_cv_train)),
                                                   test_data=TestPipeline.JobData(X[test], y[test], test, dict(kwargs_cv_test)),
                                                   save_feature_importances=save_feature_importances,
                                                   save_predictions=save_predictions)

                if not self.parallel_cv:
                    # only for unparallel processing
                    # inform children in which inner fold we are
                    # self.pipe.distribute_cv_info_to_hyperpipe_children(inner_fold_counter=fold_cnt)
                    # self.mother_inner_fold_handle(fold_cnt)

                    curr_test_fold, curr_train_fold = TestPipeline.fit_and_score(job_data)
                    list_of_score_results.append((curr_test_fold, curr_train_fold))

                    if not job_data.shall_continue:
                        break

                    TestPipeline.process_fit_results(list_of_score_results, config_item,
                                                     self.cross_validation_infos.calculate_metrics_across_folds,
                                                     self.cross_validation_infos.calculate_metrics_per_fold,
                                                     original_save_predictions, self.optimization_infos.metrics)


                    if self.optimization_infos.inner_cv_callback_functions:
                        if isinstance(self.optimization_infos.inner_cv_callback_functions, list):
                            break_cv = 0
                            for cf in self.optimization_infos.inner_cv_callback_functions:
                                if not cf.shall_continue(config_item):
                                    Logger().info(
                                        'Skip further cross validation of config because of performance constraints')
                                    break_cv += 1
                                    break
                            if break_cv > 0:
                                break
                        else:
                            if not self.optimization_infos.inner_cv_callback_functions.shall_continue(config_item):
                                Logger().info(
                                    'Skip further cross validation of config because of performance constraints')
                                break

                    fold_cnt += 1
                else:
                    folds_to_do.put(job_data)

            if self.parallel_cv:
                process_list = list()
                for w in range(nr_of_processes):
                    p = Process(target=TestPipeline.parallel_inner_cv, args=(folds_to_do, folds_done))
                    process_list.append(p)
                    p.start()

                for p in process_list:
                    p.join()

                while not folds_done.empty():
                    list_of_score_results.append(folds_done.get())

                TestPipeline.process_fit_results(list_of_score_results, config_item,
                                             self.cross_validation_infos.calculate_metrics_across_folds,
                                             self.cross_validation_infos.calculate_metrics_per_fold,
                                             original_save_predictions, self.optimization_infos.metrics)

        except Exception as e:
            if self.raise_error:
                raise e
            Logger().error(e)
            Logger().error(traceback.format_exc())
            traceback.print_exc()
            if not isinstance(e, Warning):
                config_item.config_failed = True
            config_item.config_error = str(e)
            warnings.warn('One test iteration of pipeline failed with error')

        return config_item

    class JobData:
        def __init__(self, X, y, indices, cv_kwargs):
            self.X = X
            self.y = y
            self.indices = indices
            self.cv_kwargs = cv_kwargs

    class InnerCVJob:

        def __init__(self, pipe, config, metrics, callbacks, train_data, test_data,
                     save_predictions, save_feature_importances):
            self.pipe = pipe
            self.config = config
            self.metrics = metrics
            self.callbacks = callbacks
            self.train_data = train_data
            self.test_data = test_data
            self.save_predictions = save_predictions
            self.save_feature_importances = save_feature_importances
            self.shall_continue = True

    @staticmethod
    def parallel_inner_cv(folds_to_do, folds_done):
            while True:
                try:
                    task = folds_to_do.get_nowait()
                except queue.Empty:
                    break
                else:
                    # print("Starting Inner CV from process number " + current_process().name)
                    fold_output = TestPipeline.fit_and_score(task)
                    folds_done.put(fold_output)
                    # print("Stopping Inner CV from process number " + current_process().name)

            return True

    @staticmethod
    def process_fit_results(fold_list, config_item,
                            calculate_metrics_across_folds,
                            calculate_metrics_per_fold,
                            original_save_predictions,
                            metrics):

        overall_y_pred_test = []
        overall_y_true_test = []
        overall_y_pred_train = []
        overall_y_true_train = []
        inner_fold_list = []

        for fold_cnt, (curr_test_fold, curr_train_fold) in enumerate(fold_list):
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

            # fill result tree with fold information
            inner_fold = MDBInnerFold()
            inner_fold.fold_nr = fold_cnt
            inner_fold.training = curr_train_fold
            inner_fold.validation = curr_test_fold
            # inner_fold.number_samples_training = int(len(train))
            # inner_fold.number_samples_validation = int(len(test))
            inner_fold_list.append(inner_fold)

            # save all inner folds to the tree under the config item
            config_item.inner_folds = inner_fold_list

            # if we want to have metrics across all predictions from all folds:
            if calculate_metrics_across_folds:
                # metrics across folds
                metrics_to_calculate = list(metrics)
                if 'score' in metrics_to_calculate:
                    metrics_to_calculate.remove('score')
                metrics_train = TestPipeline.calculate_metrics(overall_y_true_train,
                                                               overall_y_pred_train, metrics_to_calculate)
                metrics_test = TestPipeline.calculate_metrics(overall_y_true_test,
                                                              overall_y_pred_test, metrics_to_calculate)

                def metric_to_db_class(metric_list):
                    db_metrics = []
                    for metric_name, metric_value in metric_list.items():
                        new_metric = MDBFoldMetric(operation=FoldOperations.RAW, metric_name=metric_name,
                                                   value=metric_value)
                        db_metrics.append(new_metric)
                    return db_metrics

                db_metrics_train = metric_to_db_class(metrics_train)
                db_metrics_test = metric_to_db_class(metrics_test)

                # if we want to have metrics for each fold as well, calculate mean and std.
                if calculate_metrics_per_fold:
                    db_metrics_fold_train, db_metrics_fold_test = MDBHelper.aggregate_metrics(config_item,
                                                                                              metrics)
                    config_item.metrics_train = db_metrics_train + db_metrics_fold_train
                    config_item.metrics_test = db_metrics_test + db_metrics_fold_test
                else:
                    config_item.metrics_train = db_metrics_train
                    config_item.metrics_test = db_metrics_test

                # we needed to save the true/predicted values to calculate the metrics across folds,
                # but if the user is uninterested in it we dismiss them after the job is done
                if not original_save_predictions:
                    for inner_fold in config_item.inner_folds:
                        # Todo: What about dismissing feature importances, too?
                        inner_fold.training.y_true = []
                        inner_fold.training.y_pred = []
                        inner_fold.training.indices = []
                        inner_fold.validation.y_true = []
                        inner_fold.validation.y_pred = []
                        inner_fold.validation.indices = []

            elif calculate_metrics_per_fold:
                # calculate mean and std over all fold metrics
                config_item.metrics_train, config_item.metrics_test = MDBHelper.aggregate_metrics(config_item,
                                                                                                  metrics)

    @staticmethod
    def fit_and_score(job: InnerCVJob):

        pipe = job.pipe

        # set params to current config
        pipe.set_params(**job.config)

        # start fitting
        fit_start_time = time.time()
        pipe.fit(job.train_data.X, job.train_data.y, **job.train_data.cv_kwargs)

        # Todo: Fit Process Metrics
        # write down how long the fitting took
        # fit_duration = time.time() - fit_start_time
        # config_item.fit_duration_minutes = fit_duration

        # score test data
        curr_test_fold = TestPipeline.score(pipe, job.test_data.X, job.test_data.y, job.metrics, indices=job.test_data.indices,
                                            save_predictions=job.save_predictions,
                                            save_feature_importances=job.save_feature_importances, **job.test_data.cv_kwargs)

        # score train data
        curr_train_fold = TestPipeline.score(pipe, job.train_data.X, job.train_data.y, job.metrics,
                                             indices=job.train_data.indices,
                                             save_predictions=job.save_predictions,
                                             save_feature_importances=job.save_feature_importances,
                                             training=True, **job.train_data.cv_kwargs)


        return curr_test_fold, curr_train_fold

    @staticmethod
    def score(estimator, X, y_true, metrics, indices=[],
              save_predictions=False, save_feature_importances=False,
              calculate_metrics: bool=True, training: bool=False, **kwargs):
        """
        Uses the pipeline to predict the given data, compare it to the truth values and calculate metrics

        :param estimator: the pipeline or pipeline element for prediction
        :param X: the data for prediction
        :param y_true: the truth values for the data
        :param metrics: the metrics to be calculated
        :param indices: the indices of the given data and targets that are logged into the result tree
        :param save_predictions: if True, the predicted value array is stored in to the result tree
        :param save_feature_importances: if True, the feature importances of the estimator, if any, are stored to the result tree
        :param training: if True, all training_only pipeline steps are executed, if False they are skipped
        :param calculate_metrics: if True, calculates metrics for given data
        :return: ScoreInformation object
        """

        scoring_time_start = time.time()

        output_metrics = {}
        non_default_score_metrics = list(metrics)
        # that does not work because it is not an exact match and also reacts to e.g. f1_score
        # if 'score' in metrics:
        # so we use this:
        checklist = ['score']
        matches = set(checklist).intersection(set(non_default_score_metrics))
        if len(matches) > 0:
            # Todo: Here it is potentially slowing down!!!!!!!!!!!!!!!!
            default_score = estimator.score(X, y_true)
            output_metrics['score'] = default_score
            non_default_score_metrics.remove('score')

        if not training:
            y_pred = estimator.predict(X, **kwargs)
        else:
            # durchs Cachen kommt hier X mit zu wenig items zurück...
            X, y_true_new, kwargs_new = estimator.transform(X, y_true, **kwargs)
            if y_true_new is not None:
                y_true = y_true_new
            if kwargs_new is not None and len(kwargs_new) > 0:
                kwargs = kwargs_new
            y_pred = estimator.predict(X, training=True, **kwargs)

        f_importances = []
        if save_feature_importances:
            try:
                if hasattr(estimator._final_estimator.base_element, 'coef_'):
                    f_importances = estimator._final_estimator.base_element.coef_
                    f_importances = f_importances.tolist()
                elif hasattr(estimator._final_estimator.base_element, 'feature_importances_'):
                    f_importances = estimator._final_estimator.base_element.feature_importances_
                    f_importances = f_importances.tolist()
            except:
                f_importances = None

        # Nice to have
        # TestPipeline.plot_some_data(y_true, y_pred)

        if calculate_metrics:
            score_metrics = TestPipeline.calculate_metrics(y_true, y_pred, non_default_score_metrics)

            # add default metric
            if output_metrics:
                output_metrics = {**output_metrics, **score_metrics}
            else:
                output_metrics = score_metrics
        else:
            output_metrics = {}

        final_scoring_time = time.time() - scoring_time_start
        if save_predictions:

            probabilities = []
            if hasattr(estimator._final_estimator.base_element, 'predict_proba'):
                probabilities = estimator.predict_proba(X, training=training,  **kwargs)

                try:
                    if probabilities is not None:
                        if not len(probabilities) == 0:
                            probabilities = probabilities.tolist()
                except:
                    warnings.warn('No probabilities available.')

            if isinstance(y_pred, list):
                y_pred = np.array(y_pred)
                y_true = np.array(y_true)

            score_result_object = MDBScoreInformation(metrics=output_metrics,
                                                      score_duration=final_scoring_time,
                                                      y_pred=y_pred.tolist(), y_true=y_true.tolist(),
                                                      indices=np.asarray(indices).tolist(),
                                                      probabilities=probabilities)
            if save_feature_importances:
                score_result_object.feature_importances = f_importances
        elif save_feature_importances:
            score_result_object = MDBScoreInformation(metrics=output_metrics,
                                                      score_duration=final_scoring_time,
                                                      feature_importances=f_importances)
        else:
            score_result_object = MDBScoreInformation(metrics=output_metrics,
                                                      score_duration=final_scoring_time)
        return score_result_object

    @staticmethod
    def calculate_metrics(y_true, y_pred, metrics):
        """
        Applies all metrics to the given predicted and true values.
        The metrics are encoded via a string literal which is mapped to the according calculation function
        :param y_true: the truth values
        :type y_true: list
        :param y_pred: the predicted values
        :param metrics: list
        :return: dict of metrics
        """

        # Todo: HOW TO CHECK IF ITS REGRESSION?!
        # The following works only for classification
        # if np.ndim(y_pred) == 2:
        #     y_pred = one_hot_to_binary(y_pred)
        #     Logger().warn("test_predictions was one hot encoded => transformed to binary")
        #
        # if np.ndim(y_true) == 2:
        #     y_true = one_hot_to_binary(y_true)
        #     Logger().warn("test_y was one hot encoded => transformed to binary")

        output_metrics = {}
        if metrics:
            for metric in metrics:
                scorer = Scorer.create(metric)
                if scorer is not None:
                    scorer_value = scorer(y_true, y_pred)
                    Logger().debug(str(scorer_value))
                    output_metrics[metric] = scorer_value
                else:
                    output_metrics[metric] = np.nan

        return output_metrics


class Scorer(object):
    """
    Transforms a string literal into an callable instance of a particular metric
    """

    ELEMENT_DICTIONARY = {
        # Classification
        'matthews_corrcoef': ('sklearn.metrics', 'matthews_corrcoef', 'score'),
        'confusion_matrix': ('sklearn.metrics', 'confusion_matrix', None),
        'accuracy': ('sklearn.metrics', 'accuracy_score', 'score'),
        'f1_score': ('sklearn.metrics', 'f1_score', 'score'),
        'hamming_loss': ('sklearn.metrics', 'hamming_loss', 'error'),
        'log_loss': ('sklearn.metrics', 'log_loss', 'error'),
        'precision': ('sklearn.metrics', 'precision_score', 'score'),
        'recall': ('sklearn.metrics', 'recall_score', 'score'),
        'auc': ('sklearn.metrics', 'roc_auc_score', 'score'),
        'sensitivity': ('photonai.validation.Metrics', 'sensitivity', 'score'),
        'specificity': ('photonai.validation.Metrics', 'specificity', 'score'),
        'balanced_accuracy': ('photonai.validation.Metrics', 'balanced_accuracy', 'score'),
        'categorical_accuracy': ('photonai.validation.Metrics', 'categorical_accuracy_score', 'score'),
        'categorical_crossentropy': ('photonai.validation.Metrics', 'categorical_crossentropy', 'error'),

        # Regression
        'mean_squared_error': ('sklearn.metrics', 'mean_squared_error', 'error'),
        'mean_absolute_error': ('sklearn.metrics', 'mean_absolute_error', 'error'),
        'explained_variance': ('sklearn.metrics', 'explained_variance_score', 'score'),
        'r2': ('sklearn.metrics', 'r2_score', 'score'),
        'pearson_correlation': ('photonai.validation.Metrics', 'pearson_correlation', 'score'),
        'variance_explained':  ('photonai.validation.Metrics', 'variance_explained_score', 'score')

    }

    @classmethod
    def create(cls, metric):
        """
        Searches for the metric by name and instantiates the according calculation function
        :param metric: the name of the metric as encoded in the ELEMENT_DICTIONARY
        :type metric: str
        :return: a callable instance of the metric calculation
        """
        if metric in Scorer.ELEMENT_DICTIONARY:
            try:
                desired_class_info = Scorer.ELEMENT_DICTIONARY[metric]
                desired_class_home = desired_class_info[0]
                desired_class_name = desired_class_info[1]
                imported_module = __import__(desired_class_home, globals(),
                                             locals(), desired_class_name, 0)
                desired_class = getattr(imported_module, desired_class_name)
                scoring_method = desired_class
                return scoring_method
            except AttributeError as ae:
                Logger().error('ValueError: Could not find according class: '
                               + Scorer.ELEMENT_DICTIONARY[metric])
                raise ValueError('Could not find according class:',
                                 Scorer.ELEMENT_DICTIONARY[metric])
        else:
            Logger().error('NameError: Metric not supported right now:' + metric)
            # raise Warning('Metric not supported right now:', metric)
            return None





