import csv
import os
from collections import OrderedDict
import numpy as np


class ResultLogging:

    @staticmethod
    def write_results(results_list, config_history, filename):
        cwd = os.getcwd()
        with open(cwd + "/" + filename, 'w') as csv_file:
            keys = list(results_list[0].keys())
            keys_w_space = []
            train_test = []
            for i in range(len(keys) * 2):
                if (i % 2) == 0:
                    keys_w_space.append(keys[int(i / 2)])
                    train_test.append('train')
                else:
                    keys_w_space.append('')
                    train_test.append('test')
            config_keys = list(config_history[0].keys())
            keys_w_space += config_keys
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(keys_w_space)
            writer.writerow(train_test)
        for l in range(len(results_list)):
            with open(cwd + "/" + filename, 'a') as csv_file:
                write_this_to_csv = []
                for key1, value1 in results_list[l].items():
                    for key2, value2 in results_list[l][key1].items():
                        write_this_to_csv.append(value2)
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
        r_results = OrderedDict()
        for key, value in results.items():
            # train and test should always be alternated
            # put first element under train, second under test and so forth
            train = []
            test = []
            for i in range(len(results[key])):
                if (i % 2) == 0:
                    train.append(results[key][i])
                else:
                    test.append(results[key][i])
            # again, I know this is ugly. Any suggestions? Only confusion
            # matrix behaves differently because we don't want to calculate
            # the mean of it
            if key == 'confusion_matrix':
                r_results[key] = {'train': train, 'test': test}
            else:
                r_results[key] = {'train': np.mean(train), 'test': np.mean(test)}
                r_results[key + '_std'] = {'train': np.std(train),
                                           'test': np.std(test)}
                r_results[key + '_folds'] = {'train': train, 'test': test}

        return r_results

