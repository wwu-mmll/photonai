import unittest
import numpy as np
import warnings
import glob
from shutil import rmtree

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from photonai.helper.helper import PhotonDataHelper
from photonai.base import PipelineElement, Switch, Hyperpipe, OutputSettings
from photonai.base.photon_pipeline import PhotonPipeline
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import ShuffleSplit
from photonai.optimization import FloatRange, Categorical, IntegerRange
from photonai.optimization.hyperparameters import NumberRange
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
    from smac.facade.smac_bohb_facade import BOHB4HPO

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
                _ = SMACOptimizer()

else:
    class Smac3IntegrationTest(unittest.TestCase):

        def setUp(self):
            self.s_split = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

            self.time_limit = 20

            settings = OutputSettings(project_folder='./tmp/')

            self.smac_helper = {"data": None, "initial_runs": None}

            # Scenario object
            scenario_dict = {"run_obj": "quality",
                             "deterministic": "true",
                             "wallclock_limit": self.time_limit
                             }

            # DESIGN YOUR PIPELINE
            self.pipe = Hyperpipe('basic_svm_pipe',
                                  optimizer='smac',
                                  optimizer_params={'facade': SMAC4BO,
                                                    'scenario_dict': scenario_dict,
                                                    'rng': 42,
                                                    'smac_helper': self.smac_helper},
                                  metrics=['accuracy'],
                                  random_seed=42,
                                  best_config_metric='accuracy',
                                  inner_cv=self.s_split,
                                  verbosity=0,
                                  output_settings=settings)

        def simple_classification(self):
            dataset = fetch_olivetti_faces(download_if_missing=True)
            self.X = dataset["data"]
            self.y = dataset["target"]
            return self.X, self.y

        # integration test for simple pipeline without Switch
        def test_photon_implementation_simple(self):
            # PHOTON implementation
            self.pipe.add(PipelineElement('StandardScaler'))
            self.pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
            self.pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
                                                                 'C': FloatRange(0.001, 2, range_type='logspace')},
                                         gamma='auto')
            self.X, self.y = self.simple_classification()
            self.pipe.fit(self.X, self.y)

            # direct AUTO ML implementation
            # Build Configuration Space which defines all parameters and their ranges
            cs = ConfigurationSpace()
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
            cs.add_hyperparameter(n_components)
            kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])
            cs.add_hyperparameter(kernel)
            c = UniformFloatHyperparameter("SVC__C", 0.001, 2, log=True)
            cs.add_hyperparameter(c)

            # Scenario object
            scenario = Scenario({"run_obj": "quality",
                                 "cs": cs,
                                 "deterministic": "true",
                                 "wallclock_limit": self.time_limit,
                                 "limit_resources": False,
                                 'abort_on_first_run_crash': False
                                 })

            # Optimize, using a SMAC directly
            smac = SMAC4BO(scenario=scenario, rng=42,
                           tae_runner=self.objective_function_simple)
            _ = smac.optimize()

            runhistory_photon = self.smac_helper["data"].solver.runhistory
            runhistory_original = smac.solver.runhistory

            x_ax = range(1, min(len(runhistory_original._cost_per_config.keys()),
                                len(runhistory_photon._cost_per_config.keys()))+1)
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

            min_len = min(len(y_ax_original), len(y_ax_photon))
            self.assertLessEqual(np.max(np.abs(np.array(y_ax_original[:min_len]) -
                                               np.array(y_ax_photon[:min_len])
                                               )), 0.01)

        def objective_function_simple(self, cfg):
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            values = []

            train_indices = list(self.pipe.cross_validation.outer_folds.values())[0].train_indices
            self._validation_X, self._validation_y, _ = PhotonDataHelper.split_data(self.X, self.y,
                                                                                    kwargs=None, indices=train_indices)

            for inner_fold in list(list(self.pipe.cross_validation.inner_folds.values())[0].values()):
                sc = PipelineElement("StandardScaler", {})
                pca = PipelineElement("PCA", {}, random_state=42)
                svc = PipelineElement("SVC", {}, random_state=42, gamma='auto')
                my_pipe = PhotonPipeline([('StandardScaler', sc), ('PCA', pca), ('SVC', svc)])
                my_pipe.set_params(**cfg)
                my_pipe.fit(self._validation_X[inner_fold.train_indices, :],
                            self._validation_y[inner_fold.train_indices])
                values.append(accuracy_score(self._validation_y[inner_fold.test_indices],
                                             my_pipe.predict(self._validation_X[inner_fold.test_indices, :])
                                             )
                              )
            return 1-np.mean(values)

        # integration test for pipeline with Switch
        def test_photon_implementation_switch(self):
            # PHOTON implementation
            self.pipe.add(PipelineElement('StandardScaler'))
            self.pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
            estimator_siwtch = Switch("Estimator")
            estimator_siwtch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
                                                                        'C': FloatRange(0.5, 200)}, gamma='auto')
            estimator_siwtch += PipelineElement('RandomForestClassifier',
                                                hyperparameters={'criterion': Categorical(['gini', 'entropy']),
                                                                 'min_samples_split': IntegerRange(2, 4)
                                                                 })
            self.pipe += estimator_siwtch
            self.X, self.y = self.simple_classification()
            self.pipe.fit(self.X, self.y)

            # direct AUTO ML implementation

            # Build Configuration Space which defines all parameters and their ranges
            cs = ConfigurationSpace()
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
            cs.add_hyperparameter(n_components)

            switch = CategoricalHyperparameter("Estimator_switch", ['svc', 'rf'])
            cs.add_hyperparameter(switch)

            kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])
            cs.add_hyperparameter(kernel)
            c = UniformFloatHyperparameter("SVC__C", 0.5, 200)
            cs.add_hyperparameter(c)
            use_svc_c = InCondition(child=kernel, parent=switch, values=["svc"])
            use_svc_kernel = InCondition(child=c, parent=switch, values=["svc"])

            criterion = CategoricalHyperparameter("RandomForestClassifier__criterion", ['gini', 'entropy'])
            cs.add_hyperparameter(criterion)
            minsplit = UniformIntegerHyperparameter("RandomForestClassifier__min_samples_split", 2, 4)
            cs.add_hyperparameter(minsplit)

            use_rf_crit = InCondition(child=criterion, parent=switch, values=["rf"])
            use_rf_minsplit = InCondition(child=minsplit, parent=switch, values=["rf"])

            cs.add_conditions([use_svc_c, use_svc_kernel, use_rf_crit, use_rf_minsplit])

            # Scenario object
            scenario = Scenario({"run_obj": "quality",
                                 "cs": cs,
                                 "deterministic": "true",
                                 "wallclock_limit": self.time_limit,
                                 "limit_resources": False,
                                 'abort_on_first_run_crash': False
                                 })

            # Optimize, using a SMAC directly
            smac = SMAC4BO(scenario=scenario, rng=42,
                           tae_runner=self.objective_function_switch)
            _ = smac.optimize()

            runhistory_photon = self.smac_helper["data"].solver.runhistory
            runhistory_original = smac.solver.runhistory

            x_ax = range(1, min(len(runhistory_original._cost_per_config.keys()),
                                len(runhistory_photon._cost_per_config.keys())) + 1)
            y_ax_original = [runhistory_original._cost_per_config[tmp] for tmp in x_ax]
            y_ax_photon = [runhistory_photon._cost_per_config[tmp] for tmp in x_ax]

            min_len = min(len(y_ax_original), len(y_ax_photon))
            self.assertLessEqual(np.max(np.abs(np.array(y_ax_original[:min_len]) -
                                               np.array(y_ax_photon[:min_len])
                                               )), 0.01)

        def objective_function_switch(self, cfg):
            cfg = {k: cfg[k] for k in cfg if cfg[k]}
            values = []

            train_indices = list(self.pipe.cross_validation.outer_folds.values())[0].train_indices
            self._validation_X, self._validation_y, _ = PhotonDataHelper.split_data(self.X, self.y,
                                                                                    kwargs=None,
                                                                                    indices=train_indices)

            switch = cfg["Estimator_switch"]
            del cfg["Estimator_switch"]
            for inner_fold in list(list(self.pipe.cross_validation.inner_folds.values())[0].values()):
                sc = PipelineElement("StandardScaler", {})
                pca = PipelineElement("PCA", {}, random_state=42)
                if switch == 'svc':
                    est = PipelineElement("SVC", {}, random_state=42, gamma='auto')
                    name = 'SVC'
                else:
                    est = PipelineElement("RandomForestClassifier", {}, random_state=42)
                    name = "RandomForestClassifier"
                my_pipe = PhotonPipeline([('StandardScaler', sc), ('PCA', pca), (name, est)])
                my_pipe.set_params(**cfg)
                my_pipe.fit(self._validation_X[inner_fold.train_indices, :],
                            self._validation_y[inner_fold.train_indices])
                values.append(accuracy_score(self._validation_y[inner_fold.test_indices],
                                             my_pipe.predict(self._validation_X[inner_fold.test_indices, :])
                                             )
                              )
            return 1 - np.mean(values)

        def test_facade(self):
            config_space = ConfigurationSpace()
            n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
            config_space.add_hyperparameter(n_components)
            scenario_dict = {"run_obj": "quality",
                             "deterministic": "true",
                             "cs": config_space,
                             "wallclock_limit": 60
                             }

            with self.assertRaises(ValueError):
                SMACOptimizer(facade="SMAC4BOO", scenario_dict=scenario_dict)

            with self.assertRaises(ValueError):
                facade = SMAC4BO(scenario=Scenario(scenario_dict))
                SMACOptimizer(facade=facade, scenario_dict=scenario_dict)

            facades = ["SMAC4BO", SMAC4BO, "SMAC4AC", SMAC4AC, "SMAC4HPO", SMAC4HPO, "BOHB4HPO", BOHB4HPO]
            for facade in facades:
                SMACOptimizer(facade=facade, scenario_dict=scenario_dict)

        def test_other(self):
            with warnings.catch_warnings(record=True) as w:
                opt = SMACOptimizer(facade="SMAC4BO", intensifier_kwargs={'min_chall': 2})
                assert any("No scenario_dict for smac" in s for s in [e.message.args[0] for e in w])
                self.assertIsNotNone(opt.intensifier_kwargs)

            pipeline_elements = [PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly',
                                                                                               "sigmoid"]),
                                                                        'C': [0.6]})]
            of = lambda x: x**2
            with warnings.catch_warnings(record=True) as w:
                opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)
                assert any("PHOTONAI has detected some" in s for s in [e.message.args[0] for e in w])

            pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': FloatRange(0.1, 0.5,
                                                                                         range_type='geomspace')})]
            opt = SMACOptimizer(facade="SMAC4BO")
            with self.assertRaises(NotImplementedError):
                opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)

            pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': NumberRange(1, 3,
                                                                                         range_type='range')})]
            opt = SMACOptimizer(facade="SMAC4BO")
            with self.assertRaises(ValueError):
                opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)

        @classmethod
        def tearDownClass(cls) -> None:
            dirs = glob.glob("smac*/")
            for dir in dirs:
                rmtree(dir, ignore_errors=True)