import time
import traceback
import warnings

# import matplotlib.pyplot as plt
import numpy as np

from Helpers.TFUtilities import one_hot_to_binary
from Logging.Logger import Logger
from .ResultLogging import FoldMetrics, FoldTupel, FoldOperations, Configuration, MasterElementType


class TestPipeline(object):

    def __init__(self, pipe, specific_config, metrics):

        self.params = specific_config
        self.pipe = pipe
        self.metrics = metrics

    def calculate_cv_score(self, X, y, cv_iter):

        # needed for testing Timeboxed Random Grid Search
        # time.sleep(35)

        config_item = Configuration(MasterElementType.INNER_TRAIN, self.params)
        fold_cnt = 0

        for train, test in cv_iter:
            # why clone? removed: clone(self.pipe),
            # fit_and_predict_score = _fit_and_score(self.pipe, X, y, self.score,
            #                                        train, test, self.verbose, self.params,
            #                                        fit_params=self.fit_params,
            #                                        return_train_score=self.return_train_score,
            #                                        return_n_test_samples=True,
            #                                        return_times=True, return_parameters=True,
            #                                        error_score=self.error_score)

            try:

                fold_cnt += 1


                self.pipe.set_params(**self.params)
                fit_start_time = time.time()
                self.pipe.fit(X[train], y[train])

                # Todo: Fit Process Metrics

                fit_duration = time.time()-fit_start_time
                config_item.fit_duration = fit_duration

                # score test data
                curr_test_fold = TestPipeline.score(self.pipe, X[test], y[test], self.metrics)

                # score train data
                curr_train_fold = TestPipeline.score(self.pipe, X[train], y[train], self.metrics)

                fold_tuple_item = FoldTupel(fold_cnt)
                fold_tuple_item.test = curr_test_fold
                fold_tuple_item.train = curr_train_fold
                fold_tuple_item.number_samples_train = len(train)
                fold_tuple_item.number_samples_test = len(test)
                config_item.fold_list.append(fold_tuple_item)

            except Exception as e:
                # Todo: Logging!
                Logger().error(e)
                traceback.print_exc()
                config_item.config_failed = True
                config_item.config_error = str(e)
                warnings.warn('One test iteration of pipeline failed with error')

        # calculate mean and std
        config_item.calculate_metrics(self.metrics)

        return config_item

    @staticmethod
    def score(estimator, X, y_true, metrics):

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

        # Nice to have
        # TestPipeline.plot_some_data(y_true, y_pred)

        score_metrics = TestPipeline.calculate_metrics(y_true, y_pred, non_default_score_metrics)

        # add default metric
        if output_metrics:
            output_metrics = {**output_metrics, **score_metrics}

        final_scoring_time = time.time() - scoring_time_start
        score_result_object = FoldMetrics(output_metrics, final_scoring_time, y_predicted=y_pred, y_true=y_true)

        return score_result_object

    @staticmethod
    def calculate_metrics(y_true, y_pred, metrics):

        if np.ndim(y_pred) == 2:
            y_pred = one_hot_to_binary(y_pred)
            Logger().warn("test_predictions was one hot encoded => transformed to binary")

        if np.ndim(y_true) == 2:
            y_true = one_hot_to_binary(y_true)
            Logger().warn("test_y was one hot encoded => transformed to binary")

        output_metrics = {}
        if metrics:
            for metric in metrics:
                scorer = Scorer.create(metric)
                scorer_value = scorer(y_true, y_pred)
                output_metrics[metric] = scorer_value

        return output_metrics

    # @staticmethod
    # def plot_some_data(data, targets_true, targets_pred):
    #     ax_array = np.arange(0, data.shape[0], 1)
    #     plt.figure().clear()
    #     plt.plot(ax_array, data, ax_array, targets_true, ax_array, targets_pred)
    #     plt.title('A sample of data')
    #     plt.show()


class Scorer(object):

    ELEMENT_DICTIONARY = {
        # Classification
        'matthews_corrcoef': ('sklearn.metrics', 'matthews_corrcoef'),
        'confusion_matrix': ('sklearn.metrics', 'confusion_matrix'),
        'accuracy': ('sklearn.metrics', 'accuracy_score'),
        'f1_score': ('sklearn.metrics', 'f1_score'),
        'hamming_loss': ('sklearn.metrics', 'hamming_loss'),
        'log_loss': ('sklearn.metrics', 'log_loss'),
        'precision': ('sklearn.metrics', 'precision_score'),
        'recall': ('sklearn.metrics', 'recall_score'),
        # Regression
        'mean_squared_error': ('sklearn.metrics', 'mean_squared_error'),
        'mean_absolute_error': ('sklearn.metrics', 'mean_absolute_error'),
        'explained_variance': ('sklearn.metrics', 'explained_variance_score'),
        'r2': ('sklearn.metrics', 'r2_score'),
        'categorical_accuracy': ('Framework.Metrics','categorical_accuracy_score')
    }

    # def __init__(self, estimator, x, y_true, metrics):
    #     self.estimator = estimator
    #     self.x = x
    #     self.y_true = y_true
    #     self.metrics = metrics

    @classmethod
    def create(cls, metric):
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

    def __init__(self, metric, pipeline_elements, other_metrics):
        self.metric = metric
        self.greater_is_better = None
        self.other_metrics = other_metrics
        self.set_optimizer_metric(pipeline_elements)

    def check_metrics(self):
        if self.other_metrics:
            if self.metric not in self.other_metrics:
                self.other_metrics.append(self.metric)
        # maybe there's a better solution to this
        else:
            self.other_metrics = [self.metric]
        return self.other_metrics

    def get_optimum_config(self, tested_configs):
        list_of_config_vals = []

        for config in tested_configs:
            list_of_config_vals.append(config.get_metric(FoldOperations.MEAN, self.metric, train=False))

        if self.greater_is_better:
            # max metric
            best_config_metric_nr = np.argmax(list_of_config_vals)
        else:
            # min metric
            best_config_metric_nr = np.argmin(list_of_config_vals)
        return tested_configs[best_config_metric_nr]

    def set_optimizer_metric(self, pipeline_elements):
        if isinstance(self.metric, str):
            if self.metric in Scorer.ELEMENT_DICTIONARY:
                # for now do a simple hack and set greater_is_better
                # by looking at error/score in metric name
                metric_name = Scorer.ELEMENT_DICTIONARY[self.metric][1]
                specifier = metric_name.split('_')[-1]
                if specifier == 'score':
                    self.greater_is_better = True
                elif specifier == 'error':
                    self.greater_is_better = False
                else:
                    # Todo: better error checking?
                    Logger().error('NameError: Metric not suitable for optimizer.')
                    raise NameError('Metric not suitable for optimizer.')
            else:
                Logger().error('NameError: Specify valid metric.')
                raise NameError('Specify valid metric.')
        else:
            # if no optimizer metric was chosen, use default scoring method
            self.metric = 'score'

            last_element = pipeline_elements[-1]
            if hasattr(last_element.base_element, '_estimator_type'):
                if last_element.base_element._estimator_type == 'classifier':
                    self.greater_is_better = True
                elif (last_element.base_element._estimator_type == 'regressor'
                      or last_element.base_element._estimator_type == 'transformer'
                      or last_element.base_element._estimator_type == 'clusterer'):
                    self.greater_is_better = False
            else:
                # Todo: better error checking?
                Logger().error('NotImplementedError: ' +
                               'Last pipeline element does not specify '+
                               'whether it is a classifier, regressor, transformer or '+
                               'clusterer.')
                raise NotImplementedError('Last pipeline element does not specify '
                                          'whether it is a classifier, regressor, transformer or '
                                          'clusterer.')
