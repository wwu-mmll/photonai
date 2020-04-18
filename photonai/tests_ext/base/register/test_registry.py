# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.0
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
import warnings

warnings.filterwarnings("ignore")
import pytest
import numpy as np

from photonai.base import PhotonRegistry


#
def test_PhotonRegistry_pos_arg_erroe():  # too many args
    name = "myhype"
    with pytest.raises(TypeError):
        assert PhotonRegistry(name, 1, 2, 3)


# 1
def test_PhotonRegistry_0arg():  # too many args
    assert (
        (PhotonRegistry().custom_elements == None)
        & (PhotonRegistry().custom_elements_folder == None)
        & (PhotonRegistry().custom_elements_file == None)
    )


# 2
def test_PhotonRegistry_3arg():  # too many args
    name = "CustomElements"
    element = PhotonRegistry(name)
    assert (
        (element.custom_elements == {})
        & (element.custom_elements_folder == name)
        & (element.custom_elements_file == name + "/" + name + ".json")
    )


# 3
def test_PhotonRegistry_base_PHOTON_REGISTRIES():  # too many args
    assert PhotonRegistry.base_PHOTON_REGISTRIES == ["PhotonCore", "PhotonNeuro"]


# 4
def test_PhotonRegistry_PHOTON_REGISTRIES():  # too many args
    name = "CustomElements"
    assert PhotonRegistry.PHOTON_REGISTRIES[0:2] == ["PhotonCore", "PhotonNeuro"]


# 5
def test_PhotonRegistry_PHOTON_REGISTRIES():  # too many args
    PhotonRegistry().reset()
    assert PhotonRegistry.PHOTON_REGISTRIES[0:2] == ["PhotonCore", "PhotonNeuro"]


# 6



# 7

# 8

# 9
