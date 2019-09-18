import os
from glob import glob


def create_tests_example_script():
    examples_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('test')[0], 'examples')
    files = [f for f in glob(examples_folder + "/**/*.py", recursive=True)]
    string = """
import unittest
import warnings
from shutil import rmtree
from os.path import join
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(unittest.TestCase):
    
    def setUp(self):
        self.examples_folder = "{}"
    
    def tearDown(self):
        rmtree("./tmp/", ignore_errors=True)
        rmtree("./cache/", ignore_errors=True)
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
