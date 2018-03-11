
from pymodm import connect, MongoModel, EmbeddedMongoModel, fields


class MDBFoldMetric(EmbeddedMongoModel):
    operation = fields.CharField()
    metric_name = fields.CharField()
    value = fields.CharField()


class MDBScoreInformation(EmbeddedMongoModel):
    metrics = fields.DictField()
    score_duration = fields.IntegerField()
    y_true = fields.ListField()
    y_pred = fields.ListField()
    indices = fields.ListField()
    feature_importances = fields.ListField(blank=True)


class MDBInnerFold(EmbeddedMongoModel):
    fold_nr = fields.IntegerField()
    training = fields.EmbeddedDocumentField(MDBScoreInformation)
    validation = fields.EmbeddedDocumentField(MDBScoreInformation)


class MDBConfig(EmbeddedMongoModel):
    inner_folds = fields.EmbeddedDocumentListField(MDBInnerFold)
    fit_duration_minutes = fields.IntegerField()
    config_dict = fields.DictField()
    children_config = fields.DictField(blank=True)
    config_nr = fields.IntegerField()
    config_failed = fields.BooleanField()
    config_error = fields.CharField(blank=True)
    full_model_spec = fields.DictField()
    metrics_train = fields.EmbeddedDocumentListField(MDBFoldMetric)
    metrics_test = fields.EmbeddedDocumentListField(MDBFoldMetric)


class MDBOuterFold(EmbeddedMongoModel):
    fold_nr = fields.IntegerField()
    best_config = fields.EmbeddedDocumentField(MDBConfig)
    tested_config_list = fields.EmbeddedDocumentListField(MDBConfig)


class MDBHyperpipe(MongoModel):

    name = fields.CharField()
    outer_folds = fields.EmbeddedDocumentListField(MDBOuterFold)
    time_of_results = fields.DateTimeField()

