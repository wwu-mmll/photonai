import os
from glob import glob


def create_unit_tests():
    examples_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../examples/')
    files = [f for f in glob(examples_folder + "**/*.py", recursive=True)]
    string = """
import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(unittest.TestCase):
    
    def setUp(self):
        pass
    """

    for file in files:
        if not "__init__" in file:
            string += """
    def test_{}(self):
        exec(open("{}").read())
        """.format(os.path.basename(file).strip('.py'), file)
    script = open("examples_test.py", "w")
    script.write(string)
    script.close()


if __name__ == "__main__":
    create_unit_tests()
