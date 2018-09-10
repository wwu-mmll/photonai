import time
import traceback
import warnings

import numpy as np
from sklearn.pipeline import Pipeline

from ..photonlogger.Logger import Logger
from ..validation.ResultsDatabase import MDBHelper, MDBInnerFold, MDBScoreInformation, MDBFoldMetric, \
    FoldOperations, MDBConfig


class TestPipeline(object):
    """
        Trains and tests a sklearn pipeline for a specific hyperparameter combination with cross-validation,
        calculates metrics for each fold and averages metrics over all folds
    """

    def __init__(self, pipe: Pipeline, specific_config: dict, metrics: list, mother_inner_fold_handle,
                 raise_error: bool=False, mongo_db_settings=None, callback_function=None,
                 imbalanced_strategy=None):
        """
        Creates a new TestPipeline object
        :param pipe: The sklearn pipeline instance that shall be trained and tested
        :type pipe: Pipeline
        :param specific_config: The hyperparameter configuration to test
        :type specific_config: dict
        :param metrics: List of metrics to calculate
        :type metrics: list
        :param mother_inner_fold_handle: Function handle in order to inform the hyperpipe about current inner_fold
        :type mother_inner_fold_handle: function handle
        :param raise_error: if true, raises exception when training and testing the pipeline fails
        :type raise_error: bool
        """

        self.params = specific_config
        self.pipe = pipe
        self.metrics = metrics
        self.raise_error = raise_error
        self.mother_inner_fold_handle = mother_inner_fold_handle
        self.mongo_db_settings = mongo_db_settings
        self.callback_function = callback_function
        self.imbalanced_strategy = imbalanced_strategy

    def calculate_cv_score(self, X, y, cv_iter,
                           calculate_metrics_per_fold: bool = True,
                           calculate_metrics_across_folds: bool =False):
        """
        Iterates over cross-validation folds and trains the pipeline, then uses it for predictions.
        Calculates metrics per fold and averages them over fold.
        :param X: Training and test data
        :param y: Training and test targets
        :param cv_iter: function/array that yields train and test indices
        :param save_predictions: if true, saves the predicted values into the result tree
        :param calculate_metrics_per_fold: if True, calculates metrics on predictions particularly for each fold
        :param calculate_metrics_across_folds: if True, collects predictions from all folds and calculate metrics on whole collective
        :returns: configuration class for result tree that monitors training and test performance
        """

        # needed for testing Timeboxed Random Grid Search
        # time.sleep(35)

        config_item = MDBConfig()
        config_item.inner_folds = []
        config_item.metrics_test = []
        config_item.metrics_train = []
        fold_cnt = 0

        overall_y_pred_test = []
        overall_y_true_test = []
        overall_y_pred_train = []
        overall_y_true_train = []

        # if we want to collect the predictions, we need to save them into the tree
        original_save_predictions = self.mongo_db_settings.save_predictions
        save_predictions = bool(self.mongo_db_settings.save_predictions)
        save_feature_importances = self.mongo_db_settings.save_feature_importances
        if calculate_metrics_across_folds:
            save_predictions = True

        inner_fold_list = []
        try:

            # do inner cv
            for train, test in cv_iter:

                    # if the groups are imbalanced, and a strategy is chosen, apply it here
                    if self.imbalanced_strategy:
                        train_X, train_y = self.imbalanced_strategy.fit_sample(X[train],y[train])
                    else:
                        train_X = X[train]
                        train_y = y[train]

                    # set params to current config
                    self.pipe.set_params(**self.params)

                    # inform children in which inner fold we are
                    # self.pipe.distribute_cv_info_to_hyperpipe_children(inner_fold_counter=fold_cnt)
                    self.mother_inner_fold_handle(fold_cnt)

                    # start fitting
                    fit_start_time = time.time()
                    self.pipe.fit(train_X, train_y)

                    # Todo: Fit Process Metrics

                    # write down how long the fitting took
                    fit_duration = time.time()-fit_start_time
                    config_item.fit_duration_minutes = fit_duration

                    # score test data
                    curr_test_fold = TestPipeline.score(self.pipe, X[test], y[test], self.metrics, indices=test,
                                                        save_predictions=save_predictions,
                                                        save_feature_importances=save_feature_importances)

                    # score train data
                    curr_train_fold = TestPipeline.score(self.pipe, X[train], y[train], self.metrics, indices=train,
                                                         save_predictions=save_predictions,
                                                         save_feature_importances=save_feature_importances)

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
                    #inner_fold.number_samples_training = int(len(train))
                    #inner_fold.number_samples_validation = int(len(test))
                    inner_fold_list.append(inner_fold)

                    fold_cnt += 1

                    if self.callback_function:
                        if isinstance(self.callback_function, list):
                            break_cv = 0
                            for cf in self.callback_function:
                                if not cf.shall_continue(inner_fold_list):
                                    Logger().info('Skip further cross validation of config because of performance constraints')
                                    break_cv += 1
                                    break
                            if break_cv > 0:
                                break
                        else:
                            if not self.callback_function.shall_continue(inner_fold_list):
                                Logger().info(
                                    'Skip further cross validation of config because of performance constraints')
                                break

            # save all inner folds to the tree under the config item
            config_item.inner_folds = inner_fold_list

            # if we want to have metrics across all predictions from all folds:
            if calculate_metrics_across_folds:
                # metrics across folds
                metrics_to_calculate = list(self.metrics)
                if 'score' in metrics_to_calculate:
                    metrics_to_calculate.remove('score')
                metrics_train = TestPipeline.calculate_metrics(overall_y_true_train, overall_y_pred_train, metrics_to_calculate)
                metrics_test = TestPipeline.calculate_metrics(overall_y_true_test, overall_y_pred_test, metrics_to_calculate)

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
                                                                                              self.metrics)
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
                                                                                                  self.metrics)

        except Exception as e:
            if self.raise_error:
                raise e
            Logger().error(e)
            traceback.print_exc()
            config_item.config_failed = True
            config_item.config_error = str(e)
            warnings.warn('One test iteration of pipeline failed with error')

        return config_item

    @staticmethod
    def score(estimator, X, y_true, metrics, indices=[],
              save_predictions=False, save_feature_importances=False,
              calculate_metrics: bool=True):
        """
        Uses the pipeline to predict the given data, compare it to the truth values and calculate metrics

        :param estimator: the pipeline or pipeline element for prediction
        :param X: the data for prediction
        :param y_true: the truth values for the data
        :param metrics: the metrics to be calculated
        :param indices: the indices of the given data and targets that are logged into the result tree
        :param save_predictions: if True, the predicted value array is stored in to the result tree
        :param calculate_metrics: if True, calculates metrics for given data
        :return: ScoreInformation object
        """

        scoring_time_start = time.time()

        output_metrics = {}
        non_default_score_metrics = list(metrics)
        if 'score' in metrics:
            if hasattr(estimator, 'score'):
                # Todo: Here it is potentially slowing down!!!!!!!!!!!!!!!!
                default_score = estimator.score(X, y_true)
                output_metrics['score'] = default_score
                non_default_score_metrics.remove('score')

        y_pred = estimator.predict(X)

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
                probabilities = estimator.predict_proba(X)
                if probabilities != None:
                    if not len(probabilities) == 0:
                        probabilities = probabilities.tolist()

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
                scorer_value = scorer(y_true, y_pred)
                output_metrics[metric] = scorer_value

        return output_metrics


class Scorer(object):
    """
    Transforms a string literal into an callable instance of a particular metric
    """

    ELEMENT_DICTIONARY = {
        # Classification
        'matthews_corrcoef': ('sklearn.metrics', 'matthews_corrcoef', None),
        'confusion_matrix': ('sklearn.metrics', 'confusion_matrix', None),
        'accuracy': ('sklearn.metrics', 'accuracy_score', 'score'),
        'f1_score': ('sklearn.metrics', 'f1_score', 'score'),
        'hamming_loss': ('sklearn.metrics', 'hamming_loss', 'error'),
        'log_loss': ('sklearn.metrics', 'log_loss', 'error'),
        'precision': ('sklearn.metrics', 'precision_score', 'score'),
        'recall': ('sklearn.metrics', 'recall_score', 'score'),
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
        'pearson_correlation': ('photonai.validation.Metrics', 'pearson_correlation', None),
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
            raise NameError('Metric not supported right now:', metric)


class OptimizerMetric(object):
    """
    Manages the metric that is chosen to pick the best hyperparameter configuration.
    Automatically detects if the metric is better when the value increases or decreases.
    """

    def __init__(self, metric, pipeline_elements, other_metrics):
        self.metric = metric
        self.greater_is_better = None
        self.other_metrics = other_metrics
        self.set_optimizer_metric(pipeline_elements)

    def check_metrics(self):
        """
        Checks the metric settings for convenience.

        Check if the best config metric is included int list of metrics to be calculated.
        Check if the best config metric is set but list of metrics is empty.
        :return: validated list of metrics
        """
        if self.other_metrics:
            if self.metric not in self.other_metrics:
                self.other_metrics.append(self.metric)
        # maybe there's a better solution to this
        else:
            self.other_metrics = [self.metric]
        return self.other_metrics

    def get_optimum_config(self, tested_configs):
        """
        Looks for the best configuration according to the metric with which the configurations are compared -> best config metric
        :param tested_configs: the list of tested configurations and their performances
        :return: MDBConfiguration that has performed best
        """

        list_of_config_vals = []
        list_of_non_failed_configs = [conf for conf in tested_configs if not conf.config_failed]

        if len(list_of_non_failed_configs) == 0:
            raise Warning("No Configs found which did not fail.")
        try:
            for config in list_of_non_failed_configs:
                list_of_config_vals.append(MDBHelper.get_metric(config, FoldOperations.MEAN, self.metric, train=False))

            if self.greater_is_better:
                # max metric
                best_config_metric_nr = np.argmax(list_of_config_vals)
            else:
                # min metric
                best_config_metric_nr = np.argmin(list_of_config_vals)
            return list_of_non_failed_configs[best_config_metric_nr]
        except BaseException as e:
            Logger().error(str(e))

    def get_optimum_config_outer_folds(self, outer_folds):

        list_of_scores = list()
        for outer_fold in outer_folds:
            metrics = outer_fold.best_config.inner_folds[0].validation.metrics
            list_of_scores.append(metrics[self.metric])

        if self.greater_is_better:
            # max metric
            best_config_metric_nr = np.argmax(list_of_scores)
        else:
            # min metric
            best_config_metric_nr = np.argmin(list_of_scores)

        best_config = outer_folds[best_config_metric_nr].best_config
        best_config_mdb = MDBConfig()
        best_config_mdb.config_dict = best_config.config_dict
        best_config_mdb.children_config_ref = best_config.children_config_ref
        best_config_mdb.children_config_dict = best_config.children_config_dict
        best_config_mdb.human_readable_config = best_config.human_readable_config
        return best_config_mdb


    def set_optimizer_metric(self, pipeline_elements):
        """
        Analyse and prepare the best config metric.
        Derive if it is better when the value increases or decreases.
        :param pipeline_elements: the items of the pipeline
        """
        if isinstance(self.metric, str):
            if self.metric in Scorer.ELEMENT_DICTIONARY:
                # for now do a simple hack and set greater_is_better
                # by looking at error/score in metric name
                metric_name = Scorer.ELEMENT_DICTIONARY[self.metric][1]
                specifier = Scorer.ELEMENT_DICTIONARY[self.metric][2]
                if specifier == 'score':
                    self.greater_is_better = True
                elif specifier == 'error':
                    self.greater_is_better = False
                else:
                    # Todo: better error checking?
                    error_msg = "Metric not suitable for optimizer."
                    Logger().error(error_msg)
                    raise NameError(error_msg)
            else:
                Logger().error('NameError: Specify valid metric.')
                raise NameError('Specify valid metric.')
        else:
            # if no optimizer metric was chosen, use default scoring method
            self.metric = 'score'

            last_element = pipeline_elements[-1]
            if hasattr(last_element.base_element, '_estimator_type'):
                self.greater_is_better = True
            else:
                # Todo: better error checking?
                Logger().error('NotImplementedError: ' +
                               'Last pipeline element does not specify '+
                               'whether it is a classifier, regressor, transformer or '+
                               'clusterer.')
                raise NotImplementedError('Last pipeline element does not specify '
                                          'whether it is a classifier, regressor, transformer or '
                                          'clusterer.')
