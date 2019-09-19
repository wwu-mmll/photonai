import unittest
import os
from shutil import rmtree


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
