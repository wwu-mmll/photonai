# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")

import pytest
import numpy as np


from sklearn.metrics import fowlkes_mallows_score
from photonai.processing.metrics import Scorer
from photonai.errors import PhotonaiError, PhotonaiError

# 1
def test_Scorer_ML_TYPES():
    assert Scorer.ML_TYPES == ["Classification", "Regression", "Clustering"]


# 2
def test_Scorer_ML_TYPES_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.ML_TYPES == ["Classification", "Regression", "ClusterFOO"]


# 3
def test_Scorer_TYPES():
    assert Scorer.SCORE_TYPES == ["score", "error", "unsupervised"]


# 4
def test_Scorer_TYPES_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.SCORE_TYPES == ["score", "error", "un-superduper"]


# 5
def test_ML_SCORES():
    assert list(Scorer.ML_METRIC_METADATA.keys()) == [
        "Classification",
        "Regression",
        "Clustering",
    ]


# 6
def test_ML_SCORES_wrong():
    with pytest.raises(AssertionError):
        assert list(Scorer.ML_METRIC_METADATA.keys()) == [
            "Class",
            "Regression",
            "Clustering",
        ]


# 7
def test_count_ML_SCORES_CLSFID():
    assert (
        len(
            (Scorer.ML_METRIC_METADATA[list(Scorer.ML_METRIC_METADATA.keys())[Scorer.CLSFID]])
        )
        == 12
    )


# 8
def test_count_ML_SCORES_REGRID():
    assert (
        len(
            (Scorer.ML_METRIC_METADATA[list(Scorer.ML_METRIC_METADATA.keys())[Scorer.REGRID]])
        )
        == 7
    )


# 9
def test_count_ML_SCORES_CLSTDID():
    assert (
        len(
            (Scorer.ML_METRIC_METADATA[list(Scorer.ML_METRIC_METADATA.keys())[Scorer.CLSTID]])
        )
        == 7
    )


# 10 _1
def test_count_METRIC_COUNT():
    assert len(list(Scorer.METRIC_METADATA.keys())) == 26




# 11 _1
def test_METRIC_SIGN_POS():
    assert Scorer.metric_sign("FM") == 1

# 11 _2
def test_greater_is_better_distinction_POS():
    assert Scorer.greater_is_better_distinction("FM") == True

# 11 _0
def test_METRIC_SIGN_ZERO():
    assert Scorer.metric_sign("SC") == 0

# 11 _2
def test_greater_is_better_distinction_ZERO():
    with pytest.raises(PhotonaiError):
        assert Scorer.greater_is_better_distinction("SC") == True

# 11 _-1
def test_count_METRIC_SIGN_NEG():
    assert Scorer.metric_sign("hamming_loss") == -1


# 11 _2
def test_greater_is_better_distinction_NEG():
    assert Scorer.greater_is_better_distinction("hamming_loss") == False

# 12
def test_create_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.create("FM") == "fred"


# 13
def test_create_metric_FM():
    assert type(Scorer.create("FM")) == type(fowlkes_mallows_score)


# 14
def test_metric_is_element_type():
    assert Scorer.is_element_type(Scorer.ELEMENT_TYPES[Scorer.TRANID])

def test_metric_is_element_type_estimator():
    assert Scorer.is_element_type(Scorer.ELEMENT_TYPES[Scorer.ESTID])

def test_metric_is_element_type_error():
    with pytest.raises(PhotonaiError):
        assert Scorer.is_element_type('fred')

def test_metric_is_metric_classif():
    assert Scorer.is_metric('recall')
def test_metric_is_metric_linear():
    assert Scorer.is_metric('mean_absolute_error')
def test_metric_is_metric_cluster():
    assert Scorer.is_metric('CH')


def test_metric_is_metricerror():
    with pytest.raises(PhotonaiError):
        assert Scorer.is_metric('fred')
# 15
def test_metric_sign_FM():
    assert Scorer.metric_sign("FM") == Scorer.SCORE_SIGN[Scorer.SCORE_POSID]


# 16
def test_metric_sign_SC():
    assert Scorer.metric_sign("SC") == Scorer.SCORE_SIGN[Scorer.SCORE_ZEROID]


# 17
def test_metric_sign_bad():
    with pytest.raises(PhotonaiError):
        assert Scorer.metric_sign("fred") == Scorer.SCORE_SIGN[Scorer.SCORE_ZEROID]


# 18
def test_calculate_metric_accuracy():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ["accuracy"]
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {"accuracy": 0.75}


# 19
def test_calculate_metric_HCV():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ["HCV"]
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {"HCV": 1.0}


# 20
# clustering netric, no y_predict
def test_calculate_metric_silhouette_score():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ["SC"]
    s = Scorer()
    assert s.calculate_metrics(np.vstack((yt, yt, yp, yp)), yp, metrics) == {"SC": 0.0}


# 21
def test_calculate_metric_mean_absolute_error():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ["mean_absolute_error"]
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {"mean_absolute_error": 0.25}


# 22
def test_is_machine_learning_type_bad():
    with pytest.raises(PhotonaiError):
        assert Scorer.is_machine_learning_type("fred")


# 23
def test_is_machine_learning_type():
    assert Scorer.is_machine_learning_type("Clustering")

# 24
def test_is_is_element_type_bad():
    with pytest.raises(PhotonaiError):
        assert Scorer.is_element_type("Clustering")

# 25
def test_is_is_element_type_Transformer():
    assert Scorer.is_element_type("Transformer")

# 26
def test_is_is_element_type_Estimator():
    assert Scorer.is_element_type("Estimator")
