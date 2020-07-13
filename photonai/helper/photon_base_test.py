import os
import unittest
from shutil import rmtree

import numpy as np


class PhotonBaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if hasattr(cls, 'file'):
            cls.base_folder = os.path.dirname(os.path.abspath(cls.file))

            cls.cache_folder_path = os.path.join(cls.base_folder, "cache")
            os.makedirs(cls.cache_folder_path, exist_ok=True)

            cls.tmp_folder_path = os.path.join(cls.base_folder, "tmp")
            os.makedirs(cls.tmp_folder_path, exist_ok=True)

            cls.dask_path = os.path.join(cls.base_folder, "dask-worker-space")
            cls.photon_setup_error_path = os.path.join(cls.base_folder, "photon_setup_errors.log")

    def setUp(self) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.cache_folder_path, ignore_errors=True)
        rmtree(cls.tmp_folder_path, ignore_errors=True)
        rmtree(cls.dask_path, ignore_errors=True)
        if os.path.isfile(cls.photon_setup_error_path):
            os.remove(cls.photon_setup_error_path)


def elements_to_dict(elements):
    if isinstance(elements, dict):
        new_dict = dict()
        for name, element in elements.items():
            new_dict[name] = elements_to_dict(element)
        elements = new_dict
    elif isinstance(elements, list):
        new_list = list()
        for element in elements:
            new_list.append(elements_to_dict(element))
        elements = new_list
    elif isinstance(elements, tuple):
        new_list = list()
        for element in elements:
            new_list.append(elements_to_dict(element))
        elements = tuple(new_list)
    elif hasattr(elements, '__dict__'):
        new_dict = dict()
        elements = elements.__dict__
        for name, element in elements.items():
            new_dict[name] = elements_to_dict(element)
        elements = new_dict
    else:
        if not isinstance(elements, (str, float, int, complex, np.ndarray)):
            return None
    return elements
