import unittest
import numpy as np
import itertools
import warnings


from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt

from photonai.base import PipelineElement
from photonai.base.photon_pipeline import PhotonPipeline
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import ShuffleSplit
from photonai.base import Hyperpipe, OutputSettings
from photonai.optimization import FloatRange, Categorical, IntegerRange
from photonai.optimization.smac.smac import SMACOptimizer

try:
    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
        UniformFloatHyperparameter, UniformIntegerHyperparameter
    from ConfigSpace.conditions import InCondition
    # Import SMAC-utilities
    from smac.tae.execute_func import ExecuteTAFuncDict
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.facade.smac_ac_facade import SMAC4AC

    found = True
except ModuleNotFoundError:
    found = False


warnings.filterwarnings("ignore")

if not found:
    class SMACOptimizerWithoutRequirementsTest(unittest.TestCase):

        def test_imports(self):
            """
            Test for ModuleNotFoundError (requirements.txt).
            """
            with self.assertRaises(ModuleNotFoundError):
                from photonai.optimization.smac.smac import SMACOptimizer
                smac = SMACOptimizer()

else:
    class Smac3IntegrationTest(unittest.TestCase):

        def setUp(self):
            self.s_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

            self.time_limit = 60*2

            settings = OutputSettings(project_folder='./tmp/')

            self.smac_helper = {"data": None, "initial_runs": None}

            # Scenario object
            scenario_dict = {"run_obj": "quality",
                             "deterministic": "true",
                             "wallclock_limit": self.time_limit
                            }

            # DESIGN YOUR PIPELINE
            self.pipe = Hyperpipe('basic_svm_pipe',
                                  optimizer='smac',  # which optimizer PHOTON shall use
                                  optimizer_params={'facade': SMAC4BO,
                                                    'scenario_dict': scenario_dict,
                                                    'rng': 42,
                                                    'smac_helper': self.smac_helper},
                                  metrics=['accuracy'],
                                  random_seed = 42,
                                  best_config_metric='accuracy',
                                  inner_cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                                  verbosity=0,
                                  output_settings=settings)

        def simple_classification(self):
            dataset = fetch_olivetti_faces(download_if_missing=True)
            self.X = dataset["data"]
            self.y = dataset["target"]
            return self.X, self.y

        # def test_against_smac_initial_design(self):
        #     # PHOTON implementation
        #     self.pipe.add(PipelineElement('StandardScaler'))
        #     self.pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
        #     self.pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
        #                                                          'C': FloatRange(0.5, 200)}, gamma='auto')
        #     self.X, self.y = self.simple_classification()
        #     self.pipe.fit(self.X, self.y)
        #
        #
        #     # direct AUTO ML implementation
        #
        #     # Build Configuration Space which defines all parameters and their ranges
        #     cs = ConfigurationSpace()
        #     n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
        #     cs.add_hyperparameter(n_components)
        #     kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])
        #     cs.add_hyperparameter(kernel)
        #     c = UniformFloatHyperparameter("SVC__C", 0.5, 200)
        #     cs.add_hyperparameter(c)
        #
        #     # Scenario object
        #     scenario = Scenario({"run_obj": "quality",
        #                          "cs": cs,
        #                          "deterministic": "true",
        #                          "wallclock_limit": self.time_limit,
        #                          "limit_resources" : False
        #                          })
        #
        #     # Optimize, using a SMAC-object
        #     print("Optimizing! Depending on your machine, this might take a few minutes.")
        #     smac = SMAC4BO(scenario=scenario, rng=42,
        #                    tae_runner=self.objective_function)
        #
        #     self.helper0 = smac
        #
        #     incumbent = smac.optimize()
        #
        #     inc_value = self.objective_function(incumbent)
        #
        #
        #     runhistory_photon = self.smac_helper["data"].solver.runhistory
        #     runhistory_original = smac.solver.runhistory
        #
        #
        #     x_ax = range(1, min(len(runhistory_original._cost_per_config.keys()), len(runhistory_photon._cost_per_config.keys()))+1)
        #     y_ax_original = [runhistory_original._cost_per_config[tmp] for tmp in x_ax]
        #     y_ax_photon = [runhistory_photon._cost_per_config[tmp] for tmp in x_ax]
        #
        #     y_ax_original_inc = [min(y_ax_original[:tmp+1]) for tmp in x_ax]
        #     y_ax_photon_inc = [min(y_ax_photon[:tmp+1]) for tmp in x_ax]
        #
        #     plot = False
        #     if plot:
        #         plt.figure(figsize=(10, 7))
        #         plt.plot(x_ax, y_ax_original, 'g', label='Original')
        #         plt.plot(x_ax, y_ax_photon, 'b', label='PHOTON')
        #         plt.plot(x_ax, y_ax_photon_inc, 'r', label='PHOTON Incumbent')
        #         plt.plot(x_ax, y_ax_original_inc, 'k', label='Original Incumbent')
        #         plt.title('Photon Prove')
        #         plt.xlabel('X')
        #         plt.ylabel('Y')
        #         plt.legend(loc='best')
        #         plt.savefig("smac.png")
        #
        #
        #     def neighbours(items, fill=None):
        #         before = itertools.chain([fill], items)
        #         after = itertools.chain(items, [fill])  # You could use itertools.zip_longest() later instead.
        #         next(after)
        #         for a, b, c in zip(before, items, after):
        #             yield [value for value in (a, b, c) if value is not fill]
        #
        #     original_pairing = [sum(values)/len(values) for values in neighbours(y_ax_original)]
        #     bias_term = np.mean([abs(y_ax_original_inc[t]-y_ax_photon_inc[t]) for t in range(len(y_ax_photon_inc))])
        #     photon_pairing = [sum(values)/len(values)-bias_term for values in neighbours(y_ax_photon)]
        #     counter = 0
        #     for i in range(24):
        #         if abs(y_ax_original[i]-y_ax_photon[i]) > 0.1:
        #             counter +=1
        #     # self.assertLessEqual(counter/24, 0.05)

        def objective_function(self, cfg):
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            sc = PipelineElement("StandardScaler", {})
            pca = PipelineElement("PCA", {}, random_state=42)
            svc = PipelineElement("SVC", {}, random_state=42, gamma='auto')
            my_pipe = PhotonPipeline([('StandardScaler', sc), ('PCA', pca), ('SVC', svc)])
            my_pipe.set_params(**cfg)

            X, Xx, y, yy = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

            metric = cross_validate(my_pipe,
                                    X, y,
                                    cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                                    scoring=make_scorer(accuracy_score, greater_is_better=True)) #, scoring=my_pipe.predict)
            print("run")
            return 1-np.mean(metric["test_score"])


        def test_facade(self):
            config_space = ConfigurationSpace()
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
            config_space.add_hyperparameter(n_components)
            scenario_dict = {"run_obj": "quality",
                             "deterministic": "true",
                             "cs":config_space,
                             "wallclock_limit": 60
                             }

            with self.assertRaises(ValueError):
                SMACOptimizer(facade="SMAC4BOO", scenario_dict=scenario_dict)

            with self.assertRaises(ValueError):
                facade = SMAC4BO(scenario = Scenario(scenario_dict))
                SMACOptimizer(facade=facade, scenario_dict=scenario_dict)

            facades = ["SMAC4BO", SMAC4BO, "SMAC4AC", SMAC4AC, "SMAC4HPO", SMAC4HPO]
            for facade in facades:
                SMACOptimizer(facade=facade, scenario_dict=scenario_dict)
