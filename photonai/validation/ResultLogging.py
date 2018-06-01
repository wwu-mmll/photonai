import csv
import os
from enum import Enum
import numpy as np
from functools import total_ordering
import datetime
import plotly
import plotly.graph_objs as go
from plotly import tools

# from .ResultsDatabase import *


class FoldMetrics:

    def __init__(self, metrics, score_duration, y_true, y_predicted, indices=None, feature_importances_=None):
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

    def pretty_config_dict(self):
        output = ''
        for name, value in self.config_dict.items():
            output += name + ': ' + str(value) + '<br>'
        return output

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

    def copy_config_to_db(self, config_obj):
        db_config = MDBConfig()
        db_config.config_dict = config_obj.config_dict
        db_config.children_config = config_obj.children_configs
        db_config.config_failed = config_obj.config_failed
        db_config.config_error = config_obj.config_error
        db_config.config_nr = config_obj.config_nr
        db_config.fit_duration_minutes = config_obj.fit_duration
        # db_config.full_model_spec = config_obj.full_model_specification
        return db_config


    def copy_fold_metrics(self, metric_list):
        fold_metrics = []
        for train_metric in metric_list:
            fold_m = MDBFoldMetric()
            fold_m.operation = train_metric.operation_name
            fold_m.metric_name = train_metric.metric_name
            fold_m.value = train_metric.value
            fold_metrics.append(fold_m)
        return fold_metrics

    def copy_score_info(self, original_info, copy_all=False):

        score_info = MDBScoreInformation()
        score_info.metrics = original_info.metrics
        score_info.score_duration = original_info.score_duration
        if copy_all:
            score_info.y_true = original_info.y_true.tolist()
            score_info.y_pred = original_info.y_predicted.tolist()
            score_info.indices = original_info.indices.tolist()
            if len(original_info.feature_importances_) > 0:
                score_info.feature_importances = original_info.feature_importances_.tolist()
            else:
                score_info.feature_importances = []
        return score_info

    def write_to_db(self):

        # create hyperpipe
        hyperpipe = MDBHyperpipe()
        hyperpipe.name = self.name
        hyperpipe.time_of_results = datetime.datetime.now()

        outer_fold_list = []

        for item in self.config_list[0].fold_list:
            outer_fold = MDBOuterFold()
            outer_fold.fold_nr = item.fold_id

            if item.test:
                # copy best config and its results on test set
                best_conf_obj = item.test.config_list[0]
                outer_fold.best_config = self.copy_config_to_db(best_conf_obj)
                if best_conf_obj.fold_list:
                    outer_fold.best_config_score_test = self.copy_score_info(best_conf_obj.fold_list[0].test, copy_all=True)
                    outer_fold.best_config_score_train = self.copy_score_info(best_conf_obj.fold_list[0].train, copy_all=True)

            # copy all other configs and results on validation set
            tested_config_list = []
            for cfg in item.train.config_list:
                test_config = self.copy_config_to_db(cfg)
                test_config.metrics_train = self.copy_fold_metrics(cfg.fold_metrics_train)
                test_config.metrics_test = self.copy_fold_metrics(cfg.fold_metrics_test)

                inner_fold_list = []
                for inner_fold in cfg.fold_list:
                    db_inner_fold = MDBInnerFold()
                    db_inner_fold.fold_nr = inner_fold.fold_id
                    db_inner_fold.training = self.copy_score_info(inner_fold.train)
                    db_inner_fold.validation = self.copy_score_info(inner_fold.test)
                    inner_fold_list.append(db_inner_fold)
                test_config.inner_folds = inner_fold_list

                tested_config_list.append(test_config)
            outer_fold.tested_config_list = tested_config_list

            # save outer fold to list
            outer_fold_list.append(outer_fold)

        hyperpipe.outer_folds = outer_fold_list

        # connect
        # todo: find better place for this
        connect("mongodb://localhost:27017/photon_db")
        # save
        hyperpipe.save()

    def pull_results_from_db(self):
        connect("mongodb://localhost:27017/photon_db")
        db_entry = list(MDBHyperpipe.objects.raw({'_id': self.name}))[0]
        #self.write_db_entry_to_tree(db_entry)
        return

    # def write_db_entry_to_tree(self, db_entry):
    #
    #     outer_fold_list = list()
    #     for outer_fold_index, outer_fold in enumerate(db_entry.outer_folds):
    #
    #         for test_config_index, test_config in enumerate(outer_fold.tested_config_list):
    #
    #             inner_fold_list = list()
    #             for inner_fold_index, inner_fold in enumerate(test_config.inner_folds):
    #
    #                 self.config_list[0].fold_list[outer_fold_index].train[test_config_index][inner_fold_index].train =
    #                 self.config_list[0].fold_list[outer_fold_index].train[test_config_index][inner_fold_index].test =



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

    def get_full_dataset_predictions_for_best_config(self, outer_cv_fold: int=0, train_data=False):
        if self.me_type == MasterElementType.ROOT:
            best_config = self.get_best_config_for(outer_cv_fold)
            if best_config:
                source_folds = best_config.best_config_object_for_validation_set.fold_list
                return self._collect_predictions(source_folds, train_data)

    def get_full_dataset_predictions_for_config(self, outer_cv_fold: int=0, config_nr: int=0, train_data=False) -> dict:
        if self.me_type == MasterElementType.ROOT:
            source_folds = self.config_list[0].fold_list[outer_cv_fold].train.config_list[config_nr].fold_list
            return self._collect_predictions(source_folds, train_data)

    def _collect_predictions(self, source_folds, train_data):
        result_dict = {}
        for fold in source_folds:
            if train_data:
                source_element = fold.train
            else:
                source_element = fold.test

            for cnt in range(len(source_element.indices)):
                result_dict[source_element.indices[cnt]] = (source_element.y_true[cnt], source_element.y_predicted[cnt])

        nr_of_preds = len(result_dict)
        y_true = np.zeros((nr_of_preds,))
        y_pred = np.zeros((nr_of_preds,))

        for key, element in result_dict.items():
            y_true[key] = element[0]
            y_pred[key] = element[1]

        return y_true, y_pred

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

    def plot_config_performances_for_outer_fold(self, outer_cv_fold=0, output_filename=''):
        if not output_filename:
            output_filename = 'PHOTON_Results_' + self.name + '_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tested_configs = self.get_tested_configurations_for(outer_cv_fold=outer_cv_fold)

        tracelist = []

        col_nr = 4
        row_nr = int(np.ceil(len(tested_configs) / col_nr))

        fig = tools.make_subplots(row_nr, col_nr, shared_xaxes=True, shared_yaxes=True,
                                  subplot_titles=[item.pretty_config_dict() for item in tested_configs])

        col_cnt = 1
        row_cnt = 1

        for cfg in tested_configs:
            inner_fold_list = cfg.fold_list
            cnt = 0
            metric_list = []
            name_list = []
            text_list = []
            for fold in inner_fold_list:
                for metric_name, metric_value in fold.train.metrics.items():
                    metric_list.append(metric_value)
                    name_list.append(metric_name)
                    text_list.append('inner fold ' + str(cnt + 1))
                cnt += 1
            trace = go.Scatter(x=name_list, y=metric_list, name=str(cfg.config_dict),
                               mode="markers", text=text_list)
            if col_cnt > col_nr:
                col_cnt = 1
                row_cnt += 1
            fig.append_trace(trace, row_cnt, col_cnt)
            col_cnt += 1

        fig['layout'].update(title="", showlegend=False, width=1500)
        return plotly.offline.plot(fig, filename=output_filename)

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
                # test has to be first!
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
