from pymodm import MongoModel, EmbeddedMongoModel, fields
from enum import Enum
import numpy as np
import pickle
import uuid


class MDBFoldMetric(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    operation = fields.CharField(blank=True)
    metric_name = fields.CharField(blank=True)
    value = fields.FloatField(blank=True)


class MDBScoreInformation(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    metrics = fields.DictField(blank=True)
    score_duration = fields.IntegerField(blank=True)
    y_true = fields.ListField(blank=True)
    y_pred = fields.ListField(blank=True)
    indices = fields.ListField(blank=True)
    feature_importances = fields.ListField(blank=True)
    probabilities = fields.ListField(blank=True)
    metrics_copied_from_inner = fields.BooleanField(default=False)

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


class MDBConfig(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    photon_config_id = fields.CharField(blank=True)
    inner_folds = fields.EmbeddedDocumentListField(MDBInnerFold, default=[], blank=True)
    best_config_score = fields.EmbeddedDocumentField(MDBInnerFold, blank=True)
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


class MDBDummyResults(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    strategy = fields.CharField(blank=True)
    train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)


class MDBHyperpipeInfo(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    data = fields.DictField(blank=True)
    cross_validation = fields.DictField(blank=True)
    optimization = fields.DictField(blank=True)
    flowchart = fields.CharField(blank=True)
    metrics = fields.ListField(blank=True)
    best_config_metric = fields.CharField(blank=True)
    estimation_type = fields.CharField(blank=True)
    eval_final_performance = fields.BooleanField(blank=True)


class MDBHyperpipe(MongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    name = fields.CharField()

    permutation_id = fields.CharField()
    permutation_failed = fields.CharField(blank=True)
    permutation_test = fields.EmbeddedDocumentField(MDBPermutationResults, blank=True)

    computation_completed = fields.BooleanField(default=False)
    computation_start_time = fields.DateTimeField(blank=True)
    time_of_results = fields.DateTimeField(blank=True)

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


class FoldOperations(Enum):
    MEAN = 0
    STD = 1
    RAW = 2


class ParallelData(MongoModel):

    unprocessed_data = fields.ObjectIdField()
    processed_data = fields.ObjectIdField()


class MDBHelper:
    OPERATION_DICT = {FoldOperations.MEAN: np.mean, FoldOperations.STD: np.std}

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

        operations = [FoldOperations.MEAN, FoldOperations.STD]
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
    def get_metric(config_item, operation, metric, train=True):
        if train:
            metric = [item.value for item in config_item.metrics_train if item.operation == str(operation)
                      and item.metric_name == metric and not config_item.config_failed]
        else:
            metric = [item.value for item in config_item.metrics_test if item.operation == str(operation)
                      and item.metric_name == metric and not config_item.config_failed]
        if len(metric) == 1:
            return metric[0]
        return metric

    @staticmethod
    def load_results(filename):
        return MDBHyperpipe.from_document(pickle.load(open(filename, 'rb')))




