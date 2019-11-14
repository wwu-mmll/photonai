import unittest

from photonai.base import PipelineElement
from photonai.optimization import IntegerRange
from photonai.test.optimization.grid_search.grid_search_test import GridSearchOptimizerTest
from photonai.base.photon_pipeline import PhotonPipeline

try:
    from photonai.optimization.smac.smac3 import SMACOptimizer

    found = True
except ModuleNotFoundError:
    found = False

if found:
    class SMACOptimizerWithRequirementsTest(GridSearchOptimizerTest):

        def setUp(self):
            """
            Set up for SmacOptimizer.
            """
            self.pipeline_elements = [PipelineElement("StandardScaler"),
                                      PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 20)},
                                                      test_disabled=True),
                                      PipelineElement("SVC")]
            self.optimizer = SMACOptimizer()

        def test_all_functions_available(self):
            """
            Test existence of functions and parameters ->  .ask() .tell() .prepare()
            Has to pass cause tell() got additional attribute runtime.
            """
            pass

        def test_ask(self):
            """
            Test general functionality of .ask(). Not implemented yet.
            """
            pass

        def test_ask_advanced(self):
            """
            Test advanced functionality of .ask(). Not implemented yet.
            """
            pass

else:
    class SMACOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.smac.smac3 import SMACOptimizer




class Smac3IntegrationTest(unittest.TestCase):

    def test_against_smac(self):
        pass

    @staticmethod
    def objective_function(cfg):

        my_pipe = PhotonPipeline([('StandardScaler', StandardScaler()), ('SVC', SVC())])
        my_pipe.random_state = seed
        my_pipe.set_params(**cfg)
        my_pipe.fit(X, y)
        y_pred = my_pipe.predict(X_train)
        metric = accuracy_score(y_pred, y_true)

        return metric
