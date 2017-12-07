import csv
import os
from enum import Enum
import numpy as np
from functools import total_ordering


class FoldMetrics:

    def __init__(self, metrics, score_duration, y_true, y_predicted, indices=[], feature_importances_=[]):
        self.metrics = metrics
        self.score_duration = score_duration
        self.y_true = y_true
        self.y_predicted = y_predicted
        self.indices = indices
        self.feature_importances_ = feature_importances_

    def to_dict(self):
        base_dict = {'score_duration': self.score_duration}
        return {**base_dict, **self.metrics}

    def roc_curve(self, plot=True):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_predicted)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


class FoldTupel:

    def __init__(self, fold_id):
        self.fold_id = fold_id
        self.train = None
        self.test = None
        self.number_samples_train = 0
        self.number_samples_test = 0

    def to_dict(self):

        # Todo: Ugly bitch
        fold_name = 'inner_fold_id'
        nr_samples = 'nr_samples_inner'
        if isinstance(self.train, MasterElement):
            fold_name = 'outer_fold_id'
            nr_samples = 'nr_samples_outer'

        return {fold_name: self.fold_id,
                nr_samples + '_train': self.number_samples_train,
                nr_samples + '_test': self.number_samples_test}


class FoldOperations(Enum):
    MEAN = 0
    STD = 1
    RAW = 2


class FoldMetric:

    OPERATION_DICT = {FoldOperations.MEAN: np.mean, FoldOperations.STD: np.std}

    def __init__(self, operation_name: str, metric_name: str, value):
        self.operation_name = operation_name
        self.metric_name = metric_name
        self.value = value

    @staticmethod
    def calculate_metric(operation_name, value_list: list, **kwargs):
        if operation_name in FoldMetric.OPERATION_DICT:
            val = FoldMetric.OPERATION_DICT[operation_name](value_list, **kwargs)
        else:
            raise KeyError('Could not find function for processing metrics across folds:' + operation_name)
        return val


class Configuration:

    def __init__(self, me_type, config_dict={}):

        self.fold_list = []
        self.fit_duration = 0
        self.me_type = me_type
        self.config_nr = -1


        if self.me_type > MasterElementType.OUTER_TRAIN:
            self.config_dict = config_dict
            self.children_configs = {}
            self.config_failed = False
            self.config_error = ''
            self.full_model_specification = None

        if self.me_type == MasterElementType.OUTER_TRAIN or self.me_type == MasterElementType.INNER_TRAIN:
            self.fold_metrics_train = []
            self.fold_metrics_test = []

    def get_metric(self, operation: FoldOperations, name: str, train=True):
        if train:
            metric = [item.value for item in self.fold_metrics_train if item.operation_name == operation
                      and item.metric_name == name]
        else:
            metric = [item.value for item in self.fold_metrics_test if item.operation_name == operation
                      and item.metric_name == name]
        if len(metric) == 1:
            return metric[0]
        return metric

    def calculate_metrics(self, metrics):
        operations = [FoldOperations.MEAN, FoldOperations.STD]
        # find metric across folds
        if self.me_type == MasterElementType.INNER_TRAIN or self.me_type == MasterElementType.OUTER_TEST:
            for metric_item in metrics:
                for op in operations:
                    value_list_train = [fold.train.metrics[metric_item] for fold in self.fold_list
                                        if metric_item in fold.train.metrics]
                    if value_list_train:
                        self.fold_metrics_train.append(FoldMetric(op, metric_item, FoldMetric.calculate_metric(op, value_list_train)))
                    value_list_test = [fold.test.metrics[metric_item] for fold in self.fold_list
                                       if metric_item in fold.test.metrics]
                    if value_list_test:
                        self.fold_metrics_test.append(FoldMetric(op, metric_item, FoldMetric.calculate_metric(op, value_list_test)))
        else:
            # Todo: calculate metrics for outer folds
            pass

    def to_dict(self):
        if self.me_type == MasterElementType.ROOT:
            fit_name = "hyperparameter_search_duration"
            return {fit_name: self.fit_duration}
        else:
            fit_name = "config_fit_duration"
            output_config_dict = {fit_name: self.fit_duration, 'fail': self.config_failed,
                                  'error_message': self.config_error}
            return {**output_config_dict, **self.config_dict, **self.children_configs}


@total_ordering
class MasterElementType(Enum):
    ROOT = 0
    OUTER_TRAIN = 1
    OUTER_TEST = 2
    INNER_TRAIN = 3
    INNER_TEST = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class MasterElement:

    def __init__(self, name, me_type=MasterElementType.ROOT):
        self.name = name
        self.me_type = me_type
        self.config_list = []

    def to_dict(self):
        if self.me_type == MasterElementType.ROOT:
            return {'hyperpipe': self.name}
        else:
            return {'name': self.name}

    def print_csv_file(self, filename):

        write_to_csv_list = self.create_csv_rows(self.name)
        if len(write_to_csv_list) > 0:
            header_list = write_to_csv_list[0].keys()

            import csv
            with open(filename, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=header_list)
                writer.writeheader()
                writer.writerows(write_to_csv_list)

    def get_best_config_for(self, outer_cv_fold):
        # Todo: Try Catch? -> Photon REsult Tree Exception
        if self.me_type == MasterElementType.ROOT:
            return self.config_list[0].fold_list[outer_cv_fold].test.config_list[0]

    def get_metrics_for_inner_cv(self, outer_cv_fold: int, inner_cv_fold: int, config_nr: int, train_data: bool = False) -> dict:
        if self.me_type == MasterElementType.ROOT:
            if train_data:
                return self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list[inner_cv_fold].train.metrics
            else:
                return self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list[inner_cv_fold].test.metrics

    def get_predictions_for_inner_cv(self, outer_cv_fold: int=0, inner_cv_fold: int=0, config_nr: int=0, train_data: bool = False) -> dict:
        if self.me_type == MasterElementType.ROOT:
            if train_data:
                source_element = self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list[inner_cv_fold].train
            else:
                source_element = self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list[inner_cv_fold].test
            return {'y_true': source_element.y_true, 'y_predicted': source_element.y_predicted}

    def get_feature_importances_for_inner_cv(self, outer_cv_fold: int=0, inner_cv_fold: int=0, config_nr: int=0) -> list:
        if self.me_type == MasterElementType.ROOT:
            return self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list[inner_cv_fold].train.feature_importances_

    def _merge_metric_dicts(self, fold_list, train_data):
        result_dict = {}
        for fold in fold_list:
            if train_data:
                source_dict = fold.train.metrics
            else:
                source_dict = fold.test.metrics

        for key, value in source_dict.items():
            if key in result_dict:
                result_dict[key].append(value)
            else:
                result_dict[key] = [value]

        return result_dict

    def get_predictions_for_best_config_of_outer_cv(self, outer_cv_fold: int=0, train_data: bool = False) -> dict:
        if self.me_type == MasterElementType.ROOT:
            best_config = self.get_best_config_for(outer_cv_fold)
            if best_config:
                if train_data:
                    return {'y_true': best_config.fold_list[0].train.y_true,
                            'y_predicted': best_config.fold_list[0].train.y_predicted}
                else:
                    return {'y_true': best_config.fold_list[0].test.y_true,
                            'y_predicted': best_config.fold_list[0].test.y_predicted}

    def get_all_metrics(self, outer_cv_fold: int = 0, config_nr: int = 0, train_data: bool = False) -> dict:
        if self.me_type == MasterElementType.ROOT:
            inner_fold_list = self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list
            metric_dicts = []

            for inner_fold in inner_fold_list:
                if train_data:
                    metric_dicts.append(inner_fold.train.metrics)
                else:
                    metric_dicts.append(inner_fold.test.metrics)

            final_dict = {}
            for metric_dict in metric_dicts:
                for key, value in metric_dict.items():
                    if key in final_dict:
                        final_dict[key].append(value)
                    else:
                        final_dict[key] = []
                        final_dict[key].append(value)

            return final_dict

    def get_best_config_performance_validation_set(self, outer_cv_fold: int = 0, train_data: bool = False) -> dict:
        best_config = self.get_best_config_for(outer_cv_fold)
        if best_config:
            fold_list = best_config.best_config_object_for_validation_set.fold_list
            metrics_dict = self._merge_metric_dicts(fold_list, train_data)
            return metrics_dict

    def get_best_config_performance_test_set(self, outer_cv_fold: int = 0, train_data: bool = False) -> object:
        # Todo: Try Catch?
        if self.me_type == MasterElementType.ROOT:
            if train_data:
                return self.config_list[0].fold_list[outer_cv_fold].test.config_list[0].fold_list[0].train
            else:
                return self.config_list[0].fold_list[outer_cv_fold].test.config_list[0].fold_list[0].test

    def get_tested_configurations_for(self, outer_cv_fold):
        # Todo: Try Catch?
        if self.me_type == MasterElementType.ROOT:
            return self.config_list[0].fold_list[outer_cv_fold].train.config_list


    def create_csv_rows(self, master_element_name):

        output_lines = []
        output_dict = {'master_element': master_element_name}
        for config_item in self.config_list:
            for fold in config_item.fold_list:
                common_dict = {**output_dict, **self.to_dict(), **config_item.to_dict(), **fold.to_dict()}

                if isinstance(fold.train, MasterElement):

                    output_lines_train = fold.train.create_csv_rows(master_element_name)
                    output_lines_test = fold.test.create_csv_rows(master_element_name)
                    output_lines.extend([{**common_dict, **train_line} for train_line in output_lines_train])
                    output_lines.extend([{**common_dict, **test_line} for test_line in output_lines_test])

                elif isinstance(fold.train, FoldMetrics):
                    train_dict = {**common_dict, **fold.train.to_dict()}
                    test_dict = {**common_dict, **fold.test.to_dict()}
                    output_lines.append(train_dict)
                    output_lines.append(test_dict)

        return output_lines

    def to_dict(self):
        return {'name': self.name, 'type': str(self.me_type)}


class ResultLogging:

    @staticmethod
    def write_results(results_list, config_history, filename):
        cwd = os.getcwd()
        # write header to csv file containing all metrics (two columns for train and test) and configurations in the first row
        with open(cwd + "/" + filename, 'w') as csv_file:
            metrics = list(results_list[0].keys())
            metrics_w_space = []
            train_test = []
            for i in range(len(metrics) * 2):
                if (i % 2) == 0:
                    metrics_w_space.append(metrics[int(i / 2)])
                    train_test.append('train')
                else:
                    metrics_w_space.append('')
                    train_test.append('test')
            config_keys = list(config_history[0].keys())
            metrics_w_space += config_keys
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(metrics_w_space)
            writer.writerow(train_test)

        # write results for train and test to csv file
        for l in range(len(results_list)):
            with open(cwd + "/" + filename, 'a') as csv_file:
                write_this_to_csv = []
                for metric in results_list[l].keys():
                    if isinstance(results_list[l][metric], dict):
                        write_this_to_csv.append(results_list[l][metric]['train'])
                        write_this_to_csv.append(results_list[l][metric]['test'])
                    elif isinstance(results_list[l][metric], list):
                        write_this_to_csv.append(results_list[l][metric])
                for key, value in config_history[l].items():
                    write_this_to_csv.append(value)
                writer = csv.writer(csv_file, delimiter='\t')
                writer.writerow(write_this_to_csv)

    @staticmethod
    def write_config_to_csv(config_history, filename):
        cwd = os.getcwd()
        with open(cwd + "/" + filename, 'w') as csvfile:
            keys = list(config_history[0].keys())
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(keys)
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            for i in range(len(config_history)):
                writer.writerow(config_history[i])

    @staticmethod
    def merge_dicts(list_of_dicts):
        if list_of_dicts:
            d = OrderedDict()
            for k in list_of_dicts[0].keys():
                d[k] = {'train': list(d[k]['train'] for d in list_of_dicts),
                        'test': list(d[k]['test'] for d in list_of_dicts)}
            return d
        else:
            return list_of_dicts

    @staticmethod
    def reorder_results(results):
        # black_list = ['duration']
        r_results = OrderedDict()
        for key, value in results.items():
            # train and test should always be alternated
            # put first element under test, second under train and so forth (this is because _fit_and_score() calculates
            # score of the test set first
            # if key not in black_list :
            train = []
            test = []
            for i in range(len(results[key])):
                # Test has to be first!
                if (i % 2) == 0:
                    test.append(results[key][i])
                else:
                    train.append(results[key][i])
            # again, I know this is ugly. Any suggestions? Only confusion
            # matrix behaves differently because we don't want to calculate
            # the mean of it
            if key == 'confusion_matrix':
                r_results[key] = {'train': train, 'test': test}
            else:
                train_mean = np.mean(train)
                train_std = np.std(train)
                test_mean = np.mean(test)
                test_std = np.std(test)
                r_results[key] = {'train': train_mean, 'test': test_mean}
                r_results[key + '_std'] = {'train': train_std, 'test': test_std}
                r_results[key + '_folds'] = {'train': train, 'test': test}
            # else:
            #     r_results[key] = results[key]
        return r_results

