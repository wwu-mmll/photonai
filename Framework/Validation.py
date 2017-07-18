from sklearn.model_selection._validation import _fit_and_score
from collections import OrderedDict
from .ResultLogging import ResultLogging
import numpy as np


class TestPipeline(object):

    def __init__(self, pipe, specific_config, metrics, verbose=0,
                 fit_params={}, error_score='raise'):

        self.params = specific_config
        self.pipe = pipe
        self.metrics = metrics
        # print(self.params)

        # default
        self.return_train_score = True
        self.verbose = verbose
        self.fit_params = fit_params
        self.error_score = error_score

        self.cv_results = OrderedDict()
        self.labels = []
        self.predictions = []

    def calculate_cv_score(self, X, y, cv_iter):
        # very important todo: clone pipeline!!!!!!!!!!!!!!!
        cv_scores = []
        n_train = []
        n_test = []

        for train, test in cv_iter:
            # why clone? removed: clone(self.pipe),
            fit_and_predict_score = _fit_and_score(self.pipe, X, y, self.score,
                                                   train, test, self.verbose, self.params,
                                                   fit_params=self.fit_params,
                                                   return_train_score=self.return_train_score,
                                                   return_n_test_samples=True,
                                                   return_times=True, return_parameters=True,
                                                   error_score=self.error_score)
            n_train.append(len(train))
            n_test.append(len(test))
            # self.pipe.fit(X[train], y[train])
            # fit_and_predict_score = self.pipe.score(X[test], y[test])
            cv_scores.append(fit_and_predict_score)

        # reorder results because now train and test simply alternates in a list
        # reorder_results() puts the results under keys "train" and "test"
        # it also calculates mean of metrics and std
        self.cv_results = ResultLogging.reorder_results(self.cv_results)
        self.cv_results['n_samples'] = {'train': n_train, 'test': n_test}

        result_dict = self.cv_results
        return result_dict

    def score(self, estimator, X, y_true):
        if hasattr(estimator, 'score'):
            default_score = estimator.score(X, y_true)
        else:
            default_score = -1
        # use cv_results as class variable to get results out of
        # _predict_and_score() method
        self.cv_results.setdefault('score', []).append(default_score)
        y_pred = self.pipe.predict(X)
        self.predictions.append(y_pred)
        self.labels.append(y_true)
        if self.metrics:
            for metric in self.metrics:
                scorer = Scorer.create(metric)
                # use setdefault method of dictionary to create list under
                # specific key even in case no list exists
                self.cv_results.setdefault(metric, []).append(scorer(y_true, y_pred))
        return default_score


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
        'r2': ('sklearn.metrics', 'r2_score')
    }

    def __init__(self, estimator, X, y_true, metrics):
        self.estimator = estimator
        self.X = X
        self.y_true = y_true
        self.metrics = metrics

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
                raise ValueError('Could not find according class:',
                                 Scorer.ELEMENT_DICTIONARY[metric])
        else:
            raise NameError('Metric not supported right now:', metric)


class OptimizerMetric(object):

    def __init__(self, metric, pipeline_elements, other_metrics):
        self.metric = metric
        self.greater_is_better = None
        self.other_metrics = other_metrics
        self.set_optimizer_metric(pipeline_elements)

    def check_metrics(self):
        if not self.metric == 'score':
            if self.other_metrics:
                if self.metric not in self.other_metrics:
                    self.other_metrics.append(self.metric)
            # maybe there's a better solution to this
            else:
                self.other_metrics = [self.metric]
        return self.other_metrics

    def get_optimum_config_idx(self, performance_metrics, metric_to_optimize):
        if self.greater_is_better:
            # max metric plus min std:
            one_minus_std = np.subtract(1, performance_metrics[metric_to_optimize + '_std']['test'])
            combined_metric = np.add(performance_metrics[metric_to_optimize]['test'], one_minus_std)
            best_config_nr = np.argmax(combined_metric)
        else:
            combined_metric = np.add(performance_metrics[metric_to_optimize]['test'],
                                     performance_metrics[metric_to_optimize + '_std']['test'])
            best_config_nr = np.argmin(combined_metric)
        return best_config_nr

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
                    raise NameError('Metric not suitable for optimizer.')
            else:
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
                raise NotImplementedError('Last pipeline element does not specify '
                                          'whether it is a classifier, regressor, transformer or '
                                          'clusterer.')
