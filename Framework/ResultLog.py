import numpy as np

class FoldMetricLog():
    def __init__(self):
        self.name = "F1"
        self.fold_index = 0
        self.value_train = 0
        self.value_test = 0

class MetricLog():
    def __init__(self, name):
        self.name = name
        self.std_train = 0
        self.std_test = 0
        self.fold_metric_logs = []

    def _mean_of_metric_value(self, domain):
        sum_of_metric_values = 0
        for fml in self.fold_metric_logs:
            if domain == "test":
                sum_of_metric_values += fml.value_train
            else:
                sum_of_metric_values += fml.value_test
        mean_of_metric_values = sum_of_metric_values / self.fold_metric_logs.count()
        return mean_of_metric_values

    def mean_train(self):
        return self._mean_of_metric_value("train")

    def mean_test(self):
        return self._mean_of_metric_value("test")

    def _std_of_metric_value(self, domain):
        np.std(fold_metric_logs)

class HyperpipeConfLog():
    def __init__(self, hp_configuration, cv_level, n_test, n_train):
        self.hp_configuration = hp_configuration
        self.cv_level = cv_level
        self.n_test = n_test
        self.n_train= n_train
        self.metric_logs = []

class HyperpipeLog():
    def __init__(self, name):
        self.name = name
        self.hyperpipe_conf_logs = []

class ResultLog():
    def __init__(self, analysis_id, duration, start_time):
        self.analysis_id = analysis_id
        self.duration = duration
        self.start_time = start_time
        self.hyperpipe_logs = []
