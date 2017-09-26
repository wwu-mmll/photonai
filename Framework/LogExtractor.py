import numpy as np
import csv
class LogExtractor():

    def __init__(self, result_tree):
        self.result_tree = result_tree

    def extract_csv(self, filename):
        self.analysis_id = 'not implemented!'
        start_time = 'not implemented!'
        duration = self.result_tree.config_list[0].fit_duration
        outer_folds = self.result_tree.config_list[0].fold_list
        outer_folds_stat = self.get_outer_folds_stat(outer_folds)
        stats = Stats(self.analysis_id, duration, start_time, outer_folds_stat)
        print("{}".format(stats.description()))
        stats_array = outer_folds_stat.get_description_array()

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for i in range(0, len(stats_array[0])):
                row = []
                for j in range(0, len(stats_array)):
                    row.append(stats_array[j][i])
                csv_writer.writerow(row)

    def get_configuarions_for_train_in_outer_folds(self, outer_folds):
        config_list = outer_folds[0].train.config_list
        result_config = []
        for config in config_list:
            result_config.append(config.config_dict)
        return result_config

    def get_metrics_for_outer_folds(self, outer_folds):
        return self.get_used_metric_names_form_fold(outer_folds[0])

    def get_used_metric_names_form_fold(self, fold):
        metrics = set()
        for metric in fold.train.config_list[0].fold_metrics_train:
            metrics.add(metric.metric_name)
        return list(metrics)

    def get_train_metric_stat_for_conf(self, train_metric, train_conf, outer_folds):
        for fold in outer_folds:
            for config in fold.train.config_list:
                if config.config_dict == train_conf:
                    for metric in config.fold_metrics_train:
                        if metric.metric_name == train_metric:
                            if metric.operation_name.name == 'MEAN':
                                print("Configuration: {}, Metric: {} => Value: {}".format(train_conf, train_metric, metric.value))

    def get_outer_fold_metrics(self, outer_folds):
        metrics = []
        for metric in outer_folds[0].test.config_list[0].fold_list[0].test.metrics:
            metrics.append(metric)
        return  metrics

    def get_scores_for_outer_fold(self, metric, outer_folds):
        scores = []
        for fold in outer_folds:
            scores.append(fold.test.config_list[0].fold_list[0].test.metrics[metric])
        return scores

    def get_fold_metric_form_fold(self, metric_name, fold):
        train_value = fold.test.config_list[0].fold_list[0].train.metrics[metric_name]
        test_value = fold.test.config_list[0].fold_list[0].test.metrics[metric_name]
        return FoldMetric(metric_name, train_value, test_value)

    def get_fold_metrics_from_fold(self, fold):
        fold_metrics = []
        for metric_name in self.get_used_metric_names_form_fold(fold):
            fold_metrics.append(self.get_fold_metric_form_fold(metric_name, fold))
        return  fold_metrics

    def get_outer_fold_stat(self, fold):
        config = fold.test.config_list[0].config_dict
        fold_metrics = self.get_fold_metrics_from_fold(fold)
        return FoldStat(config, fold_metrics)

    def get_outer_folds_stat(self, outer_folds: list):
        outer_folds_stat = []
        for fold in outer_folds:
            outer_folds_stat.append(self.get_outer_fold_stat(fold))
        return OuterFoldsStat(outer_folds_stat)

class Stats():
    def __init__(self, name, duration, started_at, outer_folds_stat: list):
        self.name = name
        self.duration = duration
        self.started_at = started_at
        self.outer_folds_stat = outer_folds_stat

    def description(self):
        out = """
        Analysis ID: {0}
        Duration:    {1}
        Started at:  {2}
        """.format(self.name, self.duration, self.started_at)
        out += self.outer_folds_stat.description()
        return out

class FoldsStat():
    def __init__(self, folds_stat: list):
        self.folds_stat = folds_stat

    def description(self):
        out = ""
        for metric_name in self.get_used_metrics():
            out += ("""
            {0}:
              Test:
                Mean: {1}
                STD:  {2}
              Train:
                Mean: {3}
                STD:  {4}
            """.format(metric_name, self.get_mean_for_metric(metric_name)['mean_test'],
                       self.get_std_for_metric(metric_name)['std_test'],
                       self.get_mean_for_metric(metric_name)['mean_train'],
                       self.get_std_for_metric(metric_name)['std_train']))
        idx = 0
        for fold_stat in self.folds_stat:
            out +=("""
            
            Fold No. {0}
              Config: {1}""").format(idx, fold_stat.config)
            for metric in fold_stat.fold_metrics:
                out += """
              {0}:
                Test:  {1}
                Train: {2} """.format(metric.name, metric.test_value, metric.train_value)
            idx += 1
        return out
    def get_description_array(self):
        description_array = []
        description_array.append(self.get_description_array_headlines())
        description_array.append(self.get_description_array_folds())
        return description_array

    def get_description_array_headlines(self):
        desription_array = []
        for metric_name in self.get_used_metrics():
            # Name Metric
            desription_array.append("Name Metric")
            # Mean Test
            desription_array.append("Mean Test")
            # Std Test
            desription_array.append("STD Test")
            # Mean Train
            desription_array.append("Mean Train")
            # Std Test
            desription_array.append("STD Train")
            # Blank Line
            desription_array.append("")
        idx = 0
        for fold_stat in self.folds_stat:
            # Fold Number
            desription_array.append("Fold Number")
            # Fold Configuration
            desription_array.append("Configuration")
            for metric in fold_stat.fold_metrics:
                # Name Metric
                desription_array.append("Name Metric")
                # Test
                desription_array.append("Test")
                # Train
                desription_array.append("Train")
                # Blank Line
                desription_array.append("")
            # Blank Line
            desription_array.append("")
            idx += 1
        return desription_array

    def get_description_array_folds(self):
        description_array = []
        for metric_name in self.get_used_metrics():
            # Name Metric
            description_array.append(metric_name)
            # Mean Test
            description_array.append(self.get_mean_for_metric(metric_name)['mean_test'])
            # Std Test
            description_array.append(self.get_std_for_metric(metric_name)['std_test'])
            # Mean Train
            description_array.append(self.get_mean_for_metric(metric_name)['mean_train'])
            # Std Test
            description_array.append(self.get_std_for_metric(metric_name)['std_train'])
            # Blank Line
            description_array.append("")
        idx = 0
        for fold_stat in self.folds_stat:
            # Fold Number
            description_array.append(idx)
            # Fold Configuration
            description_array.append(fold_stat.config)
            for metric in fold_stat.fold_metrics:
                # Name Metric
                description_array.append(metric.name)
                # Test
                description_array.append(metric.test_value)
                # Train
                description_array.append(metric.train_value)
                # Blank Line
                description_array.append("")
            # Blank Line
            description_array.append("")
            idx += 1
        return description_array

    def get_mean_for_metric(self, metric):
        train = []
        test = []
        for fold in self.folds_stat:
            for f_metric in fold.fold_metrics:
                if f_metric.name == metric:
                    train.append(f_metric.train_value)
                    test.append(f_metric.test_value)
        mean_train = np.mean(train)
        mean_test = np.mean(test)
        return {'mean_train': mean_train, 'mean_test': mean_test}

    def get_std_for_metric(self, metric):
        train = []
        test = []
        for fold in self.folds_stat:
            for f_metric in fold.fold_metrics:
                if f_metric.name == metric:
                    train.append(f_metric.train_value)
                    test.append(f_metric.test_value)
        std_train = np.std(train)
        std_test = np.std(test)
        return {'std_train': std_train, 'std_test': std_test}

    def get_used_metrics(self):
        metrics = []
        for f_metrics in self.folds_stat[0].fold_metrics:
            metrics.append(f_metrics.name)
        return  metrics


class OuterFoldsStat(FoldsStat):
    def __init__(self, outer_folds_stat: list):
        super().__init__(outer_folds_stat)

class InnerFoldsStat(FoldsStat):
    def __init__(self, related_outer_fold, inner_folds_stat):
        super().__init__(inner_folds_stat)
        self.related_outer_fold = related_outer_fold

class FoldStat():
    def __init__(self, config, fold_metrics: list):
        self.config = config
        self.fold_metrics = fold_metrics

class FoldMetric():
    def __init__(self, name, train_value, test_value):
        self.name = name
        self.train_value = train_value
        self.test_value = test_value