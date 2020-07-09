import os
from glob import glob

from photonai.test.photon_base_test import PhotonBaseTest
def create_tests_example_script():
    examples_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('helper')[0], 'examples')
    files = [f for f in glob(examples_folder + "/**/*.py", recursive=True)]
    string = """
import unittest
import warnings

import os
from pathlib import Path
from os.path import join, isdir
import photonai
from photonai.test.photon_base_test import PhotonBaseTest
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(PhotonBaseTest):
    

    def setUp(self):     
        self.examples_folder = Path(os.path.dirname(os.path.realpath(photonai.__file__))).joinpath(
            'examples')        
    """

    for file in files:
        if not "__init__" in file:
            string += """
    def test_{}(self):
        exec(open(join(self.examples_folder, "{}")).read(), locals(), globals())
""".format(os.path.basename(file)[:-3], file.split('/examples/')[1])
    script = open("examples_test.py", "w")
    script.write(string)
    script.close()


if __name__ == "__main__":
    create_tests_example_script()
