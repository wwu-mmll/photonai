import os
from glob import glob
from pathlib import Path


def create_tests_example_script():
    root_path = Path(os.path.dirname(os.path.realpath(__file__))).parent
    examples_folder = root_path.joinpath('examples')
    examples_test_path = root_path.joinpath('test').joinpath('integration_tests')
    files = [f for f in glob(str(examples_folder) + "/**/*.py", recursive=True)]
    string = """

import warnings
from os.path import join, dirname, realpath
from pathlib import Path
from photonai.helper.photon_base_test import PhotonBaseTest
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(TestRunExamples, cls).setUpClass()
    
    def setUp(self):     
        self.examples_folder = Path(dirname(realpath(__file__))).parent.parent.joinpath('examples')
    """

    for file in files:
        if not "__init__" in file:
            string += """
    def test_{}(self):
        exec(open(join(self.examples_folder, "{}")).read(), locals(), globals())
""".format(os.path.basename(file)[:-3], file.split('/examples/')[1])
    script = open(os.path.join(examples_test_path, "test_examples.py"), "w")
    script.write(string)
    script.close()


if __name__ == "__main__":
    create_tests_example_script()
