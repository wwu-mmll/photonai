
from pymodm import connect, MongoModel, EmbeddedMongoModel, fields
from pymongo.errors import DocumentTooLarge
from enum import Enum
import numpy as np
from photonai.photonlogger.Logger import Logger
import pickle


class MDBFoldMetric(EmbeddedMongoModel):
    class Meta:
        final = True

    operation = fields.CharField(blank=True)
    metric_name = fields.CharField(blank=True)
    value = fields.FloatField(blank=True)


class MDBScoreInformation(EmbeddedMongoModel):
    class Meta:
        final = True

    metrics = fields.DictField(blank=True)
    score_duration = fields.IntegerField(blank=True)
    y_true = fields.ListField(blank=True)
    y_pred = fields.ListField(blank=True)
    indices = fields.ListField(blank=True)
    feature_importances = fields.ListField(blank=True)
    metrics_copied_from_inner = fields.BooleanField(default=False)


class MDBInnerFold(EmbeddedMongoModel):
    class Meta:
        final = True

    fold_nr = fields.IntegerField()
    training = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    validation = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    number_samples_training = fields.IntegerField(blank=True)
    number_samples_validation = fields.IntegerField(blank=True)


class MDBConfig(EmbeddedMongoModel):
    class Meta:
        final = True

    inner_folds = fields.EmbeddedDocumentListField(MDBInnerFold, default=[], blank=True)
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


class MDBOuterFold(EmbeddedMongoModel):
    class Meta:
        final = True

    fold_nr = fields.IntegerField(blank=True)
    best_config = fields.EmbeddedDocumentField(MDBConfig, blank=True)
    tested_config_list = fields.EmbeddedDocumentListField(MDBConfig, default=[], blank=True)

class MDBPermutationMetrics(EmbeddedMongoModel):
    class Meta:
        final = True

    metric_name = fields.CharField(blank=True)
    metric_value = fields.FloatField(blank=True)
    p_value = fields.FloatField(blank=True)
    values_permutations = fields.ListField(blank=True)

class MDBPermutationResults(EmbeddedMongoModel):
    class Meta:
        final = True

    n_perms = fields.IntegerField(blank=True)
    random_state = fields.IntegerField(blank=True)
    metrics = fields.EmbeddedDocumentListField(MDBPermutationMetrics, blank=True)


class MDBHyperpipe(MongoModel):
    class Meta:
        final = True

    name = fields.CharField() #primary_key=True
    outer_folds = fields.EmbeddedDocumentListField(MDBOuterFold, default=[], blank=True)
    time_of_results = fields.DateTimeField(blank=True)
    permutation_test = fields.EmbeddedDocumentField(MDBPermutationResults, blank=True)
    best_config = fields.EmbeddedDocumentField(MDBConfig, blank=True)
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)

class FoldOperations(Enum):
    MEAN = 0
    STD = 1
    RAW = 2

class MDBHelper():
    OPERATION_DICT = {FoldOperations.MEAN: np.mean, FoldOperations.STD: np.std}

    @staticmethod
    def aggregate_metrics(folds, metrics):
        # Check if we want to aggregate metrics over best configs of outer folds or over metrics of inner folds
        if isinstance(folds, list):
            if isinstance(folds[0], MDBOuterFold):
                folds = [fold.best_config.inner_folds[0] for fold in folds]
        else:
            # don't try to calculate anything if the config failed
            if folds.config_failed:
                return folds
            folds = folds.inner_folds

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
                                    if metric_item in fold.training.metrics]
                if value_list_train:
                    metrics_train.append(MDBFoldMetric(operation=op, metric_name=metric_item,
                                                                   value=calculate_single_metric(op, value_list_train)))
                value_list_validation = [fold.validation.metrics[metric_item] for fold in folds
                                         if metric_item in fold.validation.metrics]
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


class MongoDBWriter:
    def __init__(self, write_to_db, connect_url):
        self.write_to_db = write_to_db
        self.connect_url = connect_url

    def set_write_to_db(self, write_to_db: bool):
        self.write_to_db = write_to_db

    def set_connection(self, connection_url: str):
        self.connect_url = connection_url

    def save(self, results_tree):
        if self.write_to_db:
            connect(self.connect_url)
            Logger().debug('Write results to mongodb...')
            try:
                results_tree.save()
            except DocumentTooLarge as e:
                Logger.error('Could not save document into DB: Document too large')
        else:
            pickle.dump(results_tree.to_son(), open(results_tree.name + '_results.p', 'wb'))
