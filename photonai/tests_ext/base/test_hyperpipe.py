# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")
import pytest
import numpy as np

from photonai.base import Hyperpipe

### only @DATACLASS TESTS

#0
def test_Hyperpipe_pos_arg_erroe():
    name = 'myhype'
    with pytest.raises(ValueError):
        assert Hyperpipe(name).name == ''

# 1
def test_Hyperpipe_Data_XEqNone():
    assert Hyperpipe.Data().X == None

#2
def test_Hyperpipe_Data_Xarray():
    test_value = np.ndarray((3,), buffer=np.array([0,1,2,3]),dtype=int)
    test_value = np.vstack([test_value,test_value])
    assert (Hyperpipe.Data(X=test_value).X == test_value).all()

#3
def test_Hyperpipe_Data_y():
    test_value = [0,1,2,3]
    assert Hyperpipe.Data(y=test_value).y == test_value

#4
def test_Hyperpipe_Data_yarray():
    test_value = np.array([0,1,2,3])
    assert (Hyperpipe.Data(y=test_value).y == test_value).all()

#5
def test_CrossValidation_inner_cv():
    test_value = 3
    assert Hyperpipe.CrossValidation(test_value, 1, True, 0.3, 4, True).inner_cv == test_value

#6
def test_CrossValidation_outer_cv():
    test_value = 3
    assert Hyperpipe.CrossValidation(test_value, test_value, True, 0.3, 4, True).outer_cv == test_value
#7
def test_CrossValidation_eval_final_performance():
    test_value = True
    assert Hyperpipe.CrossValidation(test_value, 1, test_value, 0.3, 4, True).eval_final_performance == test_value

#8
def test_CrossValidation_test_size():
    test_value = 0.37
    assert Hyperpipe.CrossValidation(test_value, 1, True, test_value, 4, True).test_size == test_value

#9
def test_CrossValidation_calculate_metrics_per_fold():
    test_value = 3
    assert Hyperpipe.CrossValidation(test_value, 1, True, 0.3, test_value, True).calculate_metrics_per_fold == test_value

#10
def test_CrossValidation_calculate_metrics_across_folds():
    test_value = False
    assert Hyperpipe.CrossValidation(3, 1, True, 0.3, 4, test_value).calculate_metrics_across_folds == test_value
#11
def test_CrossValidation_outer_folds():
    test_value = None
    assert Hyperpipe.CrossValidation(3, 1, True, 0.3, 4, test_value).calculate_metrics_across_folds == test_value

#12
def test_CrossValidation_inner_folds():
    test_value = dict()
    assert Hyperpipe.CrossValidation(3, 1, True, 0.3, 4, test_value).inner_folds == test_value




