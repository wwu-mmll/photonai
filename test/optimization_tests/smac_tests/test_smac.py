import unittest
import numpy as np
import warnings
import glob
from shutil import rmtree

from sklearn.metrics import accuracy_score

from photonai.helper.helper import PhotonDataHelper
from photonai.base import PipelineElement, Switch, Hyperpipe
from photonai.base.photon_pipeline import PhotonPipeline
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import ShuffleSplit
from photonai.optimization import FloatRange, Categorical, IntegerRange
from photonai.optimization.hyperparameters import NumberRange
from photonai.optimization.smac.smac import SMACOptimizer
import photonai.optimization.smac.smac as photonai_smac

if photonai_smac.__found__:
    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
        UniformFloatHyperparameter, UniformIntegerHyperparameter
    from ConfigSpace.conditions import InCondition
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.facade.smac_ac_facade import SMAC4AC
    from smac.facade.smac_bohb_facade import BOHB4HPO


@unittest.skipIf(not photonai_smac.__found__, 'smac not available')
class Smac3IntegrationTest(unittest.TestCase):

    def setUp(self):
        self.s_split = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

        self.time_limit = 20

        self.pipe = Hyperpipe('basic_svm_pipe',
                              optimizer='smac',
                              optimizer_params={'facade': SMAC4BO,
                                                'wallclock_limit': self.time_limit,
                                                'rng': 42},
                              metrics=['accuracy'],
                              random_seed=True,
                              best_config_metric='accuracy',
                              inner_cv=self.s_split,
                              verbosity=0,
                              project_folder='./tmp/')

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
                                                             'C': FloatRange(0.5, 200)}, gamma='auto')
        self.X, self.y = self.simple_classification()
        self.pipe.fit(self.X, self.y)

        # direct AUTO ML implementation
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter("PCA__n_components", 5, 30)
        cs.add_hyperparameter(n_components)
        kernel = CategoricalHyperparameter("SVC__kernel", ["rbf", 'poly'])
        cs.add_hyperparameter(kernel)
        c = UniformFloatHyperparameter("SVC__C", 0.5, 200)
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
        smac = SMAC4BO(scenario=scenario, rng=42, tae_runner=self.objective_function_simple)
        _ = smac.optimize()

        runhistory_photon = [x.metrics_test[0].value for x in self.pipe.results.outer_folds[0].tested_config_list]
        runhistory_original = [1-x for x in list(smac.solver.runhistory._cost_per_config.values())]

        min_len = min(len(runhistory_original), len(runhistory_photon))
        np.testing.assert_almost_equal(runhistory_photon[:min_len], runhistory_original[:min_len], 1)

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
        return 1 - np.mean(values)

    def test_further_parameters(self):
        n_configurations = 4
        pipe = Hyperpipe('basic_svm_pipe',
                         optimizer='smac',
                         optimizer_params={'facade': SMAC4BO,
                                           'wallclock_limit': 1000,
                                           'ta_run_limit': n_configurations,
                                           'rng': 42},
                         metrics=['accuracy'],
                         random_seed=True,
                         best_config_metric='accuracy',
                         inner_cv=self.s_split,
                         verbosity=0,
                         project_folder='./tmp/')

        pipe.add(PipelineElement('StandardScaler'))
        pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
        pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
                                                             'C': FloatRange(0.5, 200)}, gamma='auto')
        X, y = self.simple_classification()
        pipe.fit(X, y)
        self.assertEqual(len(pipe.results.outer_folds[0].tested_config_list), n_configurations)

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
        smac = SMAC4BO(scenario=scenario, rng=42, tae_runner=self.objective_function_switch)
        _ = smac.optimize()

        runhistory_photon = [x.metrics_test[0].value for x in self.pipe.results.outer_folds[0].tested_config_list]
        runhistory_original = [1 - x for x in list(smac.solver.runhistory._cost_per_config.values())]

        min_len = min(len(runhistory_original), len(runhistory_photon))
        np.testing.assert_almost_equal(runhistory_photon[:min_len], runhistory_original[:min_len], 1)

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
        opt = SMACOptimizer(facade="SMAC4BO", intensifier_kwargs={'min_chall': 2})
        self.assertIsNotNone(opt.intensifier_kwargs)

        pipeline_elements = [PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly', "sigmoid"]),
                                                                     'C': [0.6]})]

        def of(x):
            return x ** 2
        with warnings.catch_warnings(record=True) as w:
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)
            assert any("PHOTONAI has detected some" in s for s in [e.message.args[0] for e in w])

        pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': FloatRange(0.1, 0.5, range_type='geomspace'),
                                                                     'kernel': ['rbf']})]
        opt = SMACOptimizer(facade="SMAC4BO")
        with self.assertRaises(NotImplementedError):
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)

        pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': NumberRange(1, 3, range_type='range')})]
        opt = SMACOptimizer(facade="SMAC4BO")
        with self.assertRaises(ValueError):
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)

    @classmethod
    def tearDownClass(cls) -> None:
        dirs = glob.glob("./smac3-output*/")
        for dir_name in dirs:
            rmtree(dir_name, ignore_errors=True)
