# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")
import pytest, os
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

from scipy.stats import spearmanr
from photonai.photonlogger.logger import logger
from sklearn.metrics import fowlkes_mallows_score
from photonai.processing.metrics import Scorer

# 1
def test_Scorer_ELEMENT_TYPES():
    assert Scorer.ELEMENT_TYPES == ["Classification", "Regression", "Clustering"]


# 2
def test_Scorer_ELEMENT_TYPES_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.ELEMENT_TYPES == ["Classification", "Regression", "ClusterFOO"]


# 3
def test_Scorer_TYPES():
    assert Scorer.SCORE_TYPES == ["score", "error", "unsupervised"]


# 4
def test_Scorer_TYPES_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.SCORE_TYPES == ["score", "error", "un-superduper"]


# 5
def test_ELEMENT_SCORES():
    assert list(Scorer.ELEMENT_SCORES.keys()) == [
        "Classification",
        "Regression",
        "Clustering",
    ]


# 6
def test_ELEMENT_SCORES_wrong():
    with pytest.raises(AssertionError):
        assert list(Scorer.ELEMENT_SCORES.keys()) == [
            "Class",
            "Regression",
            "Clustering",
        ]


# 7
def test_count_ELEMENT_SCORES_CLSFID():
    assert (
        len((Scorer.ELEMENT_SCORES[list(Scorer.ELEMENT_SCORES.keys())[Scorer.CLSFID]]))
        == 12
    )


# 8
def test_count_ELEMENT_SCORES_REGRID():
    assert (
        len((Scorer.ELEMENT_SCORES[list(Scorer.ELEMENT_SCORES.keys())[Scorer.REGRID]]))
        == 7
    )


# 9
def test_count_ELEMENT_SCORES_CLSTDID():
    assert (
        len((Scorer.ELEMENT_SCORES[list(Scorer.ELEMENT_SCORES.keys())[Scorer.CLSTID]]))
        == 7
    )


# 10
def test_count_METRIC_COUNT():
    assert len(list(Scorer.METRICS.keys())) == 26


# 11
def test_count_METRIC_COUNT():
    assert Scorer.metric_sign('FM')


# 12
def test_create_wrong():
    with pytest.raises(AssertionError):
        assert Scorer.create('FM')  == 'fred'

#13
def test_create_metric_FM():
    assert type(Scorer.create('FM'))  == type(fowlkes_mallows_score)

#14
def test_metric_sign_FM():
    assert Scorer.metric_sign('FM')  == Scorer.SCORE_SIGN[Scorer.SCORE_POSID]

#15
def test_removed_greater_is_better_distinction():
    with pytest.raises(AttributeError):
        assert Scorer.greater_is_better_distinction('FM') == 'fred'

#16
def test_metric_sign_SC():
     assert Scorer.metric_sign('SC') == Scorer.SCORE_SIGN[Scorer.SCORE_ZEROID]


#17
def test_metric_sign_bad():
    with pytest.raises(NameError):
        assert Scorer.metric_sign('fred') == Scorer.SCORE_SIGN[Scorer.SCORE_ZEROID]

#18
def test_calculate_metric_accuracy():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ['accuracy']
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {'accuracy': 0.75}


#19
def test_calculate_metric_HCV():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ['HCV']
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {'HCV': 1.0}


#20
# clustering netric, no y_predict
def test_calculate_metric_silhouette_score():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ['SC']
    s = Scorer()
    assert s.calculate_metrics(np.vstack((yt,yt,yp,yp)), yp, metrics) == {'SC': 0.0}


#21
def test_calculate_metric_mean_absolute_error():
    yt = [1, 1, 1, 1]
    yp = [1, 1, 1, 0]
    metrics = ['mean_absolute_error']
    s = Scorer()
    assert s.calculate_metrics(yt, yp, metrics) == {'mean_absolute_error': 0.25}




#22
