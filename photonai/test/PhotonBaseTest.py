import unittest
from shutil import rmtree


class PhotonBaseTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        rmtree("./tmp/", ignore_errors=True)
        rmtree("./cache/", ignore_errors=True)
