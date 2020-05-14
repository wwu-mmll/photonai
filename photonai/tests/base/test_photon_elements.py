# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__ntest__ = 20
__coverage__ = 0.0
__pylint__ = 0
__mppy__ = 0

import warnings

warnings.filterwarnings("ignore")
import pytest

from photonai.base import PhotonRegistry
from photonai.errors import PhotonaiError

PHOTON_pkgs = ["PhotonCore", 'PhotonCluster', "PhotonNeuro",  "CustomElements"]
PHOTON_pkgs_len = [ 122, 1, 5]
KMEANS_MD = ('KMeans', 'sklearn.cluster.KMeans', 'Estimator')
PCA_MD = ("PCA" ,  "sklearn.decomposition.PCA", "Transformer")
SM_MD =    ("SmoothImages",
      "photonai.neuro.nifti_transformations.SmoothImages", "Transformer")


#0
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





# 3 new 1add-cluster-0.0.2
def test_PhotonRegistry_base_PHOTON_REGISTRIES():  # too many args
    assert PhotonRegistry.base_PHOTON_REGISTRIES[:] == PHOTON_pkgs[:-1]


# 4
def test_PhotonRegistry_PHOTON_REGISTRIES():  # too many args
    assert PhotonRegistry.PHOTON_REGISTRIES[:] == PHOTON_pkgs[:-1]

# 2
def test_PhotonRegistry_3arg():  # too many args
    name = "CustomElements"
    element = PhotonRegistry(name)
    assert (
        (element.custom_elements == {})
        & (element.custom_elements_folder == name)
        & (element.custom_elements_file == name + "/" + name + ".json")
    )

# 5 new 1add-cluster-0.0.2
def test_PhotonRegistry_PHOTON_REGISTRIES_reset():  # too many args
    PhotonRegistry().reset()
    assert PhotonRegistry.PHOTON_REGISTRIES[0:] ==  PHOTON_pkgs[:-1]


# 6  new 1add-cluster-0.0.2
def test_PhotonRegistry_PHOTON_REGISTRIES_folder():
    f = '/photonai/base/registry'
    r = PhotonRegistry()
    r._load_custom_folder(f)
    assert r.custom_elements_folder == f


# 7   new 1add-cluster-0.0.2
def test_PhotonRegistry_PHOTON_REGISTRIES_contents():  # too many args
    f = '/photonai/base/registry'
    r = PhotonRegistry()
    r._load_custom_folder(f)
    assert r.custom_elements == {}

# 8  new 1add-cluster-0.0.2
def test_PhotonRegistry_PHOTON_REGISTRIES_activate():  # too many args
    f = '/photonai/base/registry'
    r = PhotonRegistry()
    r._load_custom_folder(f)
    assert r.activate() == None

# 9  new 1add-cluster-0.0.2
def test_PhotonRegistry_PHOTON_REGISTRIES_contents():  # too many args
    f = '/photonai/base/registry'
    r = PhotonRegistry()
    r._load_custom_folder(f)
    assert r.activate() == None

# 10  new 1add-cluster-0.0.2
def test_PhotonRegistry_load_json():  # too many args
    r = PhotonRegistry()
    assert (len(r.load_json(PHOTON_pkgs[0])) == PHOTON_pkgs_len[0] and
            len(r.load_json(PHOTON_pkgs[1])) == PHOTON_pkgs_len[1] and
            len(r.load_json(PHOTON_pkgs[2])) == PHOTON_pkgs_len[2])

# 11  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_element_metadata_KMEANS():  # too many args
    r = PhotonRegistry()
    md = r.load_json(PHOTON_pkgs[1])
    assert r.get_element_metadata('KMeans',md) == KMEANS_MD

# 12  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_element_metadata_KMEANS_ERROR():  # too many args
    r = PhotonRegistry()
    md = r.load_json(PHOTON_pkgs[1])
    with pytest.raises(PhotonaiError):
        assert r.get_element_metadata('RandomForestClassifier',md) == RF_MD

# 13  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_element_metadata_PCA():  # too many args
    r = PhotonRegistry()
    md = r.load_json(PHOTON_pkgs[0])
    assert r.get_element_metadata('PCA',md) == PCA_MD

# 14  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_element_metadata_KMEANS():  # too many args
    r = PhotonRegistry()
    md = r.load_json(PHOTON_pkgs[2])
    assert r.get_element_metadata('SmoothImages',md) == SM_MD

# 15  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_package_info():  # too many args
    r = PhotonRegistry()
    assert (len(r.get_package_info()) == sum(PHOTON_pkgs_len))

# 16  new 1add-cluster-0.0.2
def test_PhotonRegistry_get_package_info_error():  # too many args
    r = PhotonRegistry()
    assert (len(r.get_package_info('fred')) == 0 )

# 17  new 1add-cluster-0.0.2


# 18  new 1add-cluster-0.0.2


# 19  new 1add-cluster-0.0.2


# 20  new 1add-cluster-0.0.2


# 20  new 1add-cluster-0.0.2


# 20  new 1add-cluster-0.0.2


# 20  new 1add-cluster-0.0.2


# 23  new 1add-cluster-0.0.2

# 24  new 1add-cluster-0.0.2
# 25  new 1add-cluster-0.0.2
# 26  new 1add-cluster-0.0.2
# 27  new 1add-cluster-0.0.2
# 28  new 1add-cluster-0.0.2



