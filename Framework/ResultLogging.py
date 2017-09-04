import csv
import os
from collections import OrderedDict

import numpy as np
from enum import Enum
from functools import total_ordering


class OutputMetric:

    # type may be epoch performance or model performance
    def __init__(self, name, value, output_type="model_performance"):
        self.output_type = output_type
        self.name = name
        self.value = value

    def to_dict(self):
        return {self.name: self.value}


class FoldMetrics:

    def __init__(self):
        self.metrics = []
        self.score_duration = 0

    def to_dict(self):
        base_dict = {'score_duration': self.score_duration}
        for item in self.metrics:
            base_dict = {**base_dict, **item.to_dict()}
        return base_dict


class FoldTupel:

    def __init__(self, fold_id):
        self.fold_id = fold_id
        self.train = None
        self.test = None
        self.number_samples_train = 0
        self.number_samples_test = 0

    def to_dict(self):
        reference_to_train = ""
        reference_to_test = ""
        if isinstance(self.train, MasterElement):
            reference_to_train = self.train.name
            reference_to_test = self.test.name

        return {'fold_id': self.fold_id,
                'nr_samples_train': self.number_samples_train,
                'nr_samples_test': self.number_samples_test,
                'train_reference_to': reference_to_train,
                'test_reference_to': reference_to_test}


class Configuration:

    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.children_configs = {}
        self.fold_list = []
        self.fit_duration = 0
        self.config_failed = False

    def to_dict(self):
        output_config_dict = {'fit_duration': self.fit_duration}
        return {**output_config_dict, **self.config_dict, **self.children_configs}


class MasterElementType(Enum):
    ROOT = 0
    TRAIN = 1
    TEST = 2


class MasterElement:

    def __init__(self, name, me_type=MasterElementType.ROOT):
        self.name = name
        self.me_type = me_type
        self.config_list = []

        self._header_list = []
        self._first_write = True

    '''
        *****************
        CSV FILE
        ******************

        tree_structure:
        ---------------
        one
            master_element: e.g. Hyperpipe or foregoing fold
        has n
            configurations
        has n
         fold_tuples
            each of which has
                one train branch
            and
                one test branch

        --> the train and test branches can either point to another master element

        --> or they can point to one
                fold_metrics object
            which has n
                output metrics


        static_fields:
        --------------
            master_element: name of outermost element (root hyperpipe)
            name: name of current branch (e.g. root hyperpipe name + _outer_fold_1_train
            type: ROOT, TRAIN, TEST
            fit_duration: how long the fitting of the current configuration took
            fold_id: which fold number
            nr_samples_train: how many samples were used for training the model
            nr_samples_test: how many samples were used for testing the model

        dynamic fields:
        ---------------
        If type == ROOT:
            train_reference_to: name of belonging training master element
            test_reference_to: name of belonging test master element

        Else If type == TRAIN OR TEST:
            score_duration: how long the prediction of either train or test data took place
            for all metrics:
                metric name: according value
            for all hyperparameters:
                hyperparameter name: according
   '''
    def print_csv_file(self, filename):

        write_to_csv_list = self.create_csv_rows(self.name)

        import csv
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._header_list)
            writer.writeheader()
            writer.writerows(write_to_csv_list)

    def create_csv_rows(self, master_element_name, first_level_item=False):

        output_lines = []
        output_dict = {'master_element': master_element_name}
        for config_item in self.config_list:
            for fold in config_item.fold_list:
                common_dict = {**output_dict, **self.to_dict(), **config_item.to_dict(), **fold.to_dict()}

                if isinstance(fold.train, MasterElement):
                    output_lines.append(common_dict)

                    # Todo: get headers more prettily?
                    output_lines.extend(fold.train.create_csv_rows(master_element_name, first_level_item=True))
                    self._header_list = fold.train._header_list

                    output_lines.extend(fold.test.create_csv_rows(master_element_name))

                elif isinstance(fold.train, FoldMetrics):
                    train_dict = {**common_dict, **fold.train.to_dict()}
                    test_dict = {**common_dict, **fold.test.to_dict()}

                    if self._first_write and first_level_item:
                        self._first_write = False
                        self._header_list = list(train_dict.keys())

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

