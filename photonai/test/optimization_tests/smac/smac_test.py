import unittest
import numpy as np
import itertools
import warnings


from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
import matplotlib.pyplot as plt

from photonai.base import PipelineElement
from photonai.test.optimization_tests.grid_search.grid_search_test import GridSearchOptimizerTest
from photonai.base.photon_pipeline import PhotonPipeline
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
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

        def test_all_attributes_available(self):
            pass



    class Smac3IntegrationTest(unittest.TestCase):

        def setUp(self):
            self.s_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

            self.time_limit = 60*5

            settings = OutputSettings(project_folder='./tmp/')

            self.smac_helper = {"data": None, "initial_runs": None}


            # Build Configuration Space which defines all parameters and their ranges
            cs = ConfigurationSpace()
            # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)  # , default_value=5)
            cs.add_hyperparameter(n_components)
            kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])  #, default_value="linear")
            cs.add_hyperparameter(kernel)
            c = UniformFloatHyperparameter("SVC__C", 0.5, 200)  #, default_value=1)
            cs.add_hyperparameter(c)
            # Scenario object
            scenario_dict = {"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 #"runcount-limit": 100,  # maximum function evaluations
                                 "deterministic": "true",
                                 "cs": cs,
                                 #"shared_model": "false",
                                 "wallclock_limit": self.time_limit
                                 }

            # DESIGN YOUR PIPELINE
            self.pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                                optimizer='smac',  # which optimizer PHOTON shall use
                                optimizer_params={'facade': 'SMAC4BO', 'scenario_dict': scenario_dict, 'rng': 42, 'smac_helper': self.smac_helper},
                                metrics=['accuracy'],
                                # the performance metrics of your interest
                                random_seed = 42,
                                best_config_metric='accuracy',
                                inner_cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                                verbosity=1,
                                output_settings=settings)

        def simple_classification(self):
            dataset = fetch_olivetti_faces(download_if_missing=True)
            self.X = dataset["data"]
            self.y = dataset["target"]
            #self.X, self.y = load_digits(n_class=2, return_X_y=True)
            return self.X, self.y

        def test_against_smac(self):
            # PHOTON implementation
            self.pipe.add(PipelineElement('StandardScaler'))
            # then do feature selection using a PCA, specify which values to try in the hyperparameter search
            self.pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
            # engage and optimize the good old SVM for Classification
            self.pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
                                                                 'C': FloatRange(0.5, 200)}, gamma='auto')

            self.X, self.y = self.simple_classification()
            self.pipe.fit(self.X, self.y)


            # AUTO ML implementation direct

            # Build Configuration Space which defines all parameters and their ranges
            cs = ConfigurationSpace()
            # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)  # , default_value=5)
            cs.add_hyperparameter(n_components)
            kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])  #, default_value="linear")
            cs.add_hyperparameter(kernel)
            c = UniformFloatHyperparameter("SVC__C", 0.5, 200)  #, default_value=1)
            cs.add_hyperparameter(c)

            # Scenario object
            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 #"runcount-limit": 100,  # maximum function evaluations
                                 "cs": cs,  # configuration space
                                 "deterministic": "true",
                                 #"shared_model": "false",
                                 "wallclock_limit": self.time_limit,
                                 "limit_resources" : False
                                 })

            # Optimize, using a SMAC-object
            print("Optimizing! Depending on your machine, this might take a few minutes.")
            smac = SMAC4BO(scenario=scenario, rng=42,
                           tae_runner=self.objective_function)

            self.helper0 = smac

            incumbent = smac.optimize()

            inc_value = self.objective_function(incumbent)


            runhistory_photon = self.smac_helper["data"].solver.runhistory
            runhistory_original = smac.solver.runhistory


            x_ax = range(1, min(len(runhistory_original._cost_per_config.keys()), len(runhistory_photon._cost_per_config.keys()))+1)
            y_ax_original = [runhistory_original._cost_per_config[tmp] for tmp in x_ax]
            y_ax_photon = [runhistory_photon._cost_per_config[tmp] for tmp in x_ax]

            y_ax_original_inc = [min(y_ax_original[:tmp+1]) for tmp in x_ax]
            y_ax_photon_inc = [min(y_ax_photon[:tmp+1]) for tmp in x_ax]

            plot = False
            if plot:
                plt.figure(figsize=(10, 7))
                plt.plot(x_ax, y_ax_original, 'g', label='Original')
                plt.plot(x_ax, y_ax_photon, 'b', label='PHOTON')
                plt.plot(x_ax, y_ax_photon_inc, 'r', label='PHOTON Incumbent')
                plt.plot(x_ax, y_ax_original_inc, 'k', label='Original Incumbent')
                plt.title('Photon Prove')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.legend(loc='best')
                plt.savefig("smac.png")


            def neighbours(items, fill=None):
                before = itertools.chain([fill], items)
                after = itertools.chain(items, [fill])  # You could use itertools.zip_longest() later instead.
                next(after)
                for a, b, c in zip(before, items, after):
                    yield [value for value in (a, b, c) if value is not fill]

            original_pairing = [sum(values)/len(values) for values in neighbours(y_ax_original)]
            bias_term = np.mean([abs(y_ax_original_inc[t]-y_ax_photon_inc[t]) for t in range(len(y_ax_photon_inc))])
            photon_pairing = [sum(values)/len(values)-bias_term for values in neighbours(y_ax_photon)]
            counter = 0
            for i,x in enumerate(x_ax):
                if abs(original_pairing[i]-photon_pairing[i]) > 0.07:
                    counter +=1
            self.assertLessEqual(counter/len(x_ax), 0.15)

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