import os
from glob import glob


def create_tests_example_script():
    examples_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('test')[0], 'examples')
    files = [f for f in glob(examples_folder + "/**/*.py", recursive=True)]
    string = """
import unittest
import warnings

from os.path import join, isdir
from photonai.test.PhotonBaseTest import PhotonBaseTest
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(PhotonBaseTest):
    
    def setUp(self):
        self.examples_folder = "../examples"
        if not isdir(self.examples_folder):
            self.examples_folder = "../../examples"
    """.format(examples_folder)

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
