import pickle
import uuid
from enum import Enum

import numpy as np
from pymodm import MongoModel, EmbeddedMongoModel, fields


class MetricHelper:

    def get_train_metric(self, name="", operation="mean"):
        return MDBHelper.get_metric(self.metrics_train, name, operation)

    def get_test_metric(self, name="", operation="mean"):
        return MDBHelper.get_metric(self.metrics_test, name, operation)

    def get_train_metric_dict(self):
        return MDBHelper.get_dict_from_metric_list(self.metrics_train)

    def get_test_metric_dict(self):
        return MDBHelper.get_dict_from_metric_list(self.metrics_test)


class MDBScoreInformation(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    metrics = fields.DictField(blank=True)
    score_duration = fields.IntegerField(blank=True)
    y_true = fields.ListField(blank=True)
    y_pred = fields.ListField(blank=True)
    indices = fields.ListField(blank=True)
    probabilities = fields.ListField(blank=True)
    metrics_copied_from_inner = fields.BooleanField(default=False)

    def save_memory(self):
        self.y_true = list()
        self.y_pred = list()
        self.indices = list()
        self.probabilities = list()

    def __str__(self):
        return str(self.metrics)


class MDBInnerFold(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    fold_nr = fields.IntegerField()
    training = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    validation = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    number_samples_training = fields.IntegerField(blank=True)
    number_samples_validation = fields.IntegerField(blank=True)
    time_monitor = fields.DictField(blank=True)
    feature_importances = fields.ListField(blank=True)
    learning_curves = fields.ListField(blank=True)


class MDBFoldMetric(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    operation = fields.CharField(blank=True, default='')
    metric_name = fields.CharField(blank=True)
    value = fields.FloatField(blank=True)

    def __str__(self):
        return "__".join([self.metric_name, self.operation, str(self.value)])


class MDBConfig(EmbeddedMongoModel, MetricHelper):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    photon_config_id = fields.CharField(blank=True)
    inner_folds = fields.EmbeddedDocumentListField(MDBInnerFold, default=[], blank=True)
    best_config_score = fields.EmbeddedDocumentField(MDBInnerFold, blank=True)
    computation_start_time = fields.DateTimeField(blank=True)
    computation_end_time = fields.DateTimeField(blank=True)
    fit_duration_minutes = fields.IntegerField(blank=True)
    pipe_name = fields.CharField(blank=True)
    config_dict = fields.DictField(blank=True)
    children_config_dict = fields.DictField(blank=True)
    children_config_ref = fields.ListField(default=[], blank=True)
    # best_config_ref_to_train_item = fields.CharField(blank=True)
    config_nr = fields.IntegerField(blank=True)
    config_failed = fields.BooleanField(default=False)
    config_error = fields.CharField(blank=True)
    full_model_spec = fields.DictField(blank=True)
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    human_readable_config = fields.DictField(blank=True)

    def set_photon_id(self):
        self.photon_config_id = str(uuid.uuid4())

    def decrease_memory(self):
        for fold in self.inner_folds:
            fold.training.save_memory()
            fold.validation.save_memory()
            fold.feature_importances = []
            fold.time_monitor = {}


class MDBOuterFold(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    fold_nr = fields.IntegerField(blank=True)
    best_config = fields.EmbeddedDocumentField(MDBConfig, blank=True)
    tested_config_list = fields.EmbeddedDocumentListField(MDBConfig, default=[], blank=True)
    number_samples_test = fields.IntegerField(blank=True)
    class_distribution_test = fields.DictField(blank=True, default={})
    class_distribution_validation = fields.DictField(blank=True, default={})
    number_samples_validation = fields.IntegerField(blank=True)
    dummy_results = fields.EmbeddedDocumentField(MDBInnerFold, blank=True)

    def get_optimum_config(self, metric, maximize_metric, dict_filter=None, fold_operation="mean"):
        """
        Looks for the best configuration according to the metric with which the configurations are compared
        :param tested_configs: the list of tested configurations and their performances
        :return: MDBConfiguration that has performed best
        """

        list_of_non_failed_configs = [conf for conf in self.tested_config_list if not conf.config_failed]

        # filter configs by key value pair (e.g. estimators__estimator == "SVC")
        if dict_filter is not None:
            list_of_non_failed_configs = [conf for conf in list_of_non_failed_configs
                                          if dict_filter[0] in conf.config_dict
                                          and conf.config_dict[dict_filter[0]] == dict_filter[1]]

        if len(list_of_non_failed_configs) == 0:
            raise Warning("No Configs found which did not fail.")

        if len(list_of_non_failed_configs) == 1:
            best_config_outer_fold = list_of_non_failed_configs[0]
        else:
            list_of_config_vals = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]

            if maximize_metric:
                # max metric
                best_config_metric_nr = np.argmax(list_of_config_vals)
            else:
                # min metric
                best_config_metric_nr = np.argmin(list_of_config_vals)

            best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold


class MDBPermutationMetrics(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    metric_name = fields.CharField(blank=True)
    metric_value = fields.FloatField(blank=True)
    p_value = fields.FloatField(blank=True)
    values_permutations = fields.ListField(blank=True)


class MDBPermutationResults(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    n_perms = fields.IntegerField(blank=True)
    n_perms_done = fields.IntegerField(blank=True)
    random_state = fields.IntegerField(blank=True)
    metrics = fields.EmbeddedDocumentListField(MDBPermutationMetrics, blank=True)


class MDBDummyResults(EmbeddedMongoModel, MetricHelper):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    strategy = fields.CharField(blank=True)
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)


class MDBHyperpipeInfo(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    data = fields.DictField(blank=True)
    cross_validation = fields.DictField(blank=True)
    optimization = fields.DictField(blank=True)
    elements = fields.DictField(blank=True)
    metrics = fields.ListField(blank=True)
    best_config_metric = fields.CharField(blank=True)
    maximize_best_config_metric = fields.BooleanField(blank=True)
    estimation_type = fields.CharField(blank=True)
    eval_final_performance = fields.BooleanField(blank=True)
    # todo: deprecated!!! delete in later versions.
    flowchart = fields.CharField(blank=True)


class MDBHyperpipe(MongoModel, MetricHelper):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    name = fields.CharField()
    version = fields.CharField()
    output_folder = fields.CharField(blank=True)

    permutation_id = fields.CharField(blank=True)
    permutation_failed = fields.CharField(blank=True)
    permutation_test = fields.EmbeddedDocumentField(MDBPermutationResults, blank=True)

    computation_completed = fields.BooleanField(default=False)
    computation_start_time = fields.DateTimeField(blank=True)
    computation_end_time = fields.DateTimeField(blank=True)

    outer_folds = fields.EmbeddedDocumentListField(MDBOuterFold, default=[], blank=True)
    best_config = fields.EmbeddedDocumentField(MDBConfig, blank=True)
    best_config_feature_importances = fields.ListField(blank=True)
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)

    hyperpipe_info = fields.EmbeddedDocumentField(MDBHyperpipeInfo)

    # dummy estimator
    dummy_estimator = fields.EmbeddedDocumentField(MDBDummyResults, blank=True)

    # stuff for wizard connection
    user_id = fields.CharField(blank=True)
    wizard_object_id = fields.ObjectIdField(blank=True)
    wizard_system_name = fields.CharField(blank=True)


class ParallelData(MongoModel):

    unprocessed_data = fields.ObjectIdField()
    processed_data = fields.ObjectIdField()


class MDBHelper:
    OPERATION_DICT = {"mean": np.mean, "std": np.std}

    @staticmethod
    def get_metric(metric_list, name="", operation=""):
        if name and operation:
            metric = [i for i in metric_list if i.metric_name == name and i.operation == operation]
            if len(metric) == 0:
                return None
            if len(metric) == 1:
                return metric[0].value
            else:
                raise KeyError("Found multiple metrics with same operation and name.")
        elif not name and operation:
            metric_list = {i.metric_name: i.value for i in metric_list if i.operation == operation}
            return metric_list
        else:
            raise ValueError("Can get metric(s) either by passing name and operation (raw, mean, std),"
                             "or operation only. No arguments are given. ")
            return None

    @staticmethod
    def get_dict_from_metric_list(metric_list):
        best_config_metrics = {}
        for train_metric in metric_list:
            if train_metric.metric_name not in best_config_metrics:
                best_config_metrics[train_metric.metric_name] = {}
            best_config_metrics[train_metric.metric_name][train_metric.operation] = np.round(train_metric.value, 6)
        return best_config_metrics

    @staticmethod
    def aggregate_metrics_for_outer_folds(outer_folds, metrics):
        folds = [fold.best_config.best_config_score for fold in outer_folds]
        return MDBHelper._aggregate_metrics(folds, metrics)

    @staticmethod
    def aggregate_metrics_for_inner_folds(inner_folds, metrics):
        return MDBHelper._aggregate_metrics(inner_folds, metrics)

    @staticmethod
    def _aggregate_metrics(folds, metrics):

        def calculate_single_metric(operation_name, value_list: list, **kwargs):
            if operation_name in MDBHelper.OPERATION_DICT:
                val = MDBHelper.OPERATION_DICT[operation_name](value_list, **kwargs)
            else:
                raise KeyError('Could not find function for processing metrics across folds:' + operation_name)
            return val

        operations = MDBHelper.OPERATION_DICT.keys()
        metrics_train = []
        metrics_test = []
        for metric_item in metrics:
            for op in operations:
                value_list_train = [fold.training.metrics[metric_item] for fold in folds
                                    if fold.training is not None and metric_item in fold.training.metrics]
                if value_list_train:
                    metrics_train.append(MDBFoldMetric(operation=op, metric_name=metric_item,
                                                       value=calculate_single_metric(op, value_list_train)))

                value_list_validation = [fold.validation.metrics[metric_item] for fold in folds
                                         if fold.validation is not None and metric_item in fold.validation.metrics]
                if value_list_validation:
                    metrics_test.append(MDBFoldMetric(operation=op, metric_name=metric_item,
                                                      value=calculate_single_metric(op, value_list_validation)))
        return metrics_train, metrics_test

    @staticmethod
    def load_results(filename):
        return MDBHyperpipe.from_document(pickle.load(open(filename, 'rb')))
