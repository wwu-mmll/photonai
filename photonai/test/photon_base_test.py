import os
import unittest
from shutil import rmtree

import numpy as np


class PhotonBaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_folder = os.path.dirname(os.path.abspath(__file__))
        cls.cache_folder_path = os.path.join(cls.base_folder, "cache")
        cls.tmp_folder_path = os.path.join(cls.base_folder, "tmp")
        cls.dask_path = os.path.join(cls.base_folder, "dask-worker-space")

    def setUp(self) -> None:
        os.makedirs(self.cache_folder_path, exist_ok=True)
        os.makedirs(self.tmp_folder_path, exist_ok=True)

    def tearDown(self) -> None:
        rmtree(self.cache_folder_path, ignore_errors=True)
        rmtree(self.tmp_folder_path, ignore_errors=True)
        rmtree(self.dask_path, ignore_errors=True)


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
