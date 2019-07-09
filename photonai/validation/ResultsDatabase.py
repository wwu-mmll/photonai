
from pymodm import connect, MongoModel, EmbeddedMongoModel, fields
from pymongo.errors import DocumentTooLarge
from enum import Enum
import numpy as np
import pandas as pd
from ..photonlogger.Logger import Logger
import pickle
import pprint
from prettytable import PrettyTable


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


class MDBInnerFold(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    fold_nr = fields.IntegerField()
    training = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    validation = fields.EmbeddedDocumentField(MDBScoreInformation, blank=True)
    number_samples_training = fields.IntegerField(blank=True)
    number_samples_validation = fields.IntegerField(blank=True)


class MDBConfig(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

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


class DummyResults(EmbeddedMongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    strategy = fields.CharField(blank=True)
    train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)


class MDBHyperpipe(MongoModel):
    class Meta:
        final = True
        connection_alias = 'photon_core'

    name = fields.CharField()
    permutation_id = fields.CharField()
    permutation_failed = fields.CharField(blank=True)
    computation_completed = fields.BooleanField(default=False)
    computation_start_time = fields.DateTimeField(blank=True)
    best_config_metric = fields.CharField()
    eval_final_performance = fields.BooleanField(default=True)
    outer_folds = fields.EmbeddedDocumentListField(MDBOuterFold, default=[], blank=True)
    time_of_results = fields.DateTimeField(blank=True)
    permutation_test = fields.EmbeddedDocumentField(MDBPermutationResults, blank=True)
    best_config = fields.EmbeddedDocumentField(MDBConfig, blank=True)
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric, default=[], blank=True)

    # dummy estimator
    dummy_estimator = fields.EmbeddedDocumentField(DummyResults, blank=True)

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


class MongoDBWriter:
    def __init__(self, save_settings):
        self.save_settings = save_settings

    def save(self, results_tree):
        if self.save_settings.mongodb_connect_url:
            connect(self.save_settings.mongodb_connect_url, alias='photon_core')
            Logger().debug('Write results to mongodb...')
            try:
                results_tree.save()
            except DocumentTooLarge as e:
                Logger.error('Could not save document into MongoDB: Document too large')
                # try to reduce the amount of configs saved
                # if len(results_tree.outer_folds[0].tested_config_list) > 100:
                #     for outer_fold in results_tree.outer_folds:
                #         metrics_configs = [outer_fold.tested_configlist

        if self.save_settings.local_file:
            try:
                file_opened = open(self.save_settings.local_file, 'wb')
                pickle.dump(results_tree.to_son(), file_opened)
                file_opened.close()
            except OSError as e:
                Logger().error("Could not write results to local file")
                Logger().error(str(e))

        if self.save_settings.summary_filename:
            self.write_summary(results_tree)

        self.write_predictions_file(results_tree)

    def write_predictions_file(self, result_tree):
        if self.save_settings.save_predictions or self.save_settings.save_best_config_predictions:

            fold_nr_array = []
            y_pred_array = []
            y_true_array = []
            indices_array = []

            def concat_array(a1, a2):
                if len(a1) == 0:
                    return a2
                else:
                    return np.concatenate((a1, a2))

            for outer_fold in result_tree.outer_folds:
                score_info = outer_fold.best_config.inner_folds[0].validation
                y_pred_array = concat_array(y_pred_array, score_info.y_pred)
                y_true_array = concat_array(y_true_array, score_info.y_true)
                indices_array = concat_array(indices_array, score_info.indices)
                fold_nr_array = concat_array(fold_nr_array, np.ones((len(score_info.y_true),)) * outer_fold.fold_nr)

            save_df = pd.DataFrame(data={'indices': indices_array, 'fold': fold_nr_array,
                                         'y_pred': y_pred_array, 'y_true': y_true_array})
            save_df.to_csv(self.save_settings.predictions_filename)

    def write_summary(self, result_tree):

        pp = pprint.PrettyPrinter(indent=4)

        text_list = []
        intro_text = """
PHOTON RESULT SUMMARY
-------------------------------------------------------------------

ANALYSIS NAME: {}
BEST CONFIG METRIC: {}
TIME OF RESULT: {}
        
        """.format(result_tree.name, result_tree.best_config_metric, result_tree.time_of_results)
        text_list.append(intro_text)

        if result_tree.dummy_estimator:
            dummy_text = """
-------------------------------------------------------------------
BASELINE - DUMMY ESTIMATOR
(always predict mean or most frequent target)
   
strategy: {}     

            """.format(result_tree.dummy_estimator.strategy)
            text_list.append(dummy_text)
            train_metrics = self.get_dict_from_metric_list(result_tree.dummy_estimator.test)
            text_list.append(self.print_table_for_performance_overview(train_metrics, "TEST"))
            train_metrics = self.get_dict_from_metric_list(result_tree.dummy_estimator.train)
            text_list.append(self.print_table_for_performance_overview(train_metrics, "TRAINING"))


        if result_tree.best_config:
            text_list.append("""
            
-------------------------------------------------------------------
OVERALL BEST CONFIG: 
{}            
            """.format(pp.pformat(result_tree.best_config.human_readable_config)))

        text_list.append("""
MEAN AND STD FOR ALL OUTER FOLD PERFORMANCES        
        """)

        train_metrics = self.get_dict_from_metric_list(result_tree.metrics_test)
        text_list.append(self.print_table_for_performance_overview(train_metrics, "TEST"))
        train_metrics = self.get_dict_from_metric_list(result_tree.metrics_train)
        text_list.append(self.print_table_for_performance_overview(train_metrics, "TRAINING"))

        for outer_fold in result_tree.outer_folds:
            text_list.append(self.print_outer_fold(outer_fold))

        final_text = ''.join(text_list)

        try:
            text_file = open(self.save_settings.summary_filename, "w")
            text_file.write(final_text)
            text_file.close()
            Logger().info("Saved results to summary file.")
        except OSError as e:
            Logger().error("Could not write summary file")
            Logger().error(str(e))

    def get_dict_from_metric_list(self, metric_list):
        best_config_metrics = {}
        for train_metric in metric_list:
            if not train_metric.metric_name in best_config_metrics:
                best_config_metrics[train_metric.metric_name] = {}
            operation_strip = train_metric.operation.split(".")[1]
            best_config_metrics[train_metric.metric_name][operation_strip] = np.round(train_metric.value, 6)
        return best_config_metrics

    def print_table_for_performance_overview(self, metric_dict, header):
        x = PrettyTable()
        x.field_names = ["Metric Name", "MEAN", "STD"]
        for element_key, element_dict in metric_dict.items():
            x.add_row([element_key, element_dict["MEAN"], element_dict["STD"]])

        text = """
{}:
{}
                """.format(header, str(x))

        return text

    def print_outer_fold(self, outer_fold):

        pp = pprint.PrettyPrinter(indent=4)
        outer_fold_text = []

        if outer_fold.best_config is not None:
            outer_fold_text.append("""
-------------------------------------------------------------------
OUTER FOLD {}
-------------------------------------------------------------------
Best Config:
{}

Number of samples training {}
Class distribution training {}
Number of samples test {}
Class distribution test {}
            
            """.format(outer_fold.fold_nr, pp.pformat(outer_fold.best_config.human_readable_config),
                       outer_fold.best_config.inner_folds[0].number_samples_training,
                       outer_fold.class_distribution_validation,
                       outer_fold.best_config.inner_folds[0].number_samples_validation,
                       outer_fold.class_distribution_test))

            if outer_fold.best_config.config_failed:
                outer_fold_text.append("""
Config Failed: {}            
    """.format(outer_fold.best_config.config_error))

            else:
                x = PrettyTable()
                x.field_names = ["Metric Name", "Train Value", "Test Value"]
                metrics_train = outer_fold.best_config.inner_folds[0].training.metrics
                metrics_test = outer_fold.best_config.inner_folds[0].validation.metrics

                for element_key, element_value in metrics_train.items():
                    x.add_row([element_key, np.round(element_value, 6), np.round(metrics_test[element_key],6)])
                outer_fold_text.append("""
PERFORMANCE:
{}



                """.format(str(x)))

        return ''.join(outer_fold_text)




