import unittest
import numpy as np
import warnings

from sklearn.metrics import accuracy_score

from photonai.helper.helper import PhotonDataHelper
from photonai.base import PipelineElement, Hyperpipe
from photonai.base.photon_pipeline import PhotonPipeline
from photonai.optimization import FloatRange, Categorical, IntegerRange
from photonai.optimization.hyperparameters import NumberRange
from photonai.optimization import NevergradOptimizer
import photonai.optimization.nevergrad.nevergrad as photonai_ng

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import ShuffleSplit

warnings.filterwarnings("ignore")

if photonai_ng.__found__:
    import nevergrad as ng


@unittest.skipIf(not photonai_ng.__found__, 'nevergrad not available')
class NevergradIntegrationTest(unittest.TestCase):

    def setUp(self):
        self.s_split = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

        self.time_limit = 20

        # DESIGN YOUR PIPELINE
        self.pipe = Hyperpipe('basic_svm_pipe',
                              optimizer='nevergrad',
                              optimizer_params={'facade': ng.optimizers.NGO,
                                                'n_configurations': 20,
                                                'rng': 42},
                              metrics=['accuracy'],
                              best_config_metric='accuracy',
                              inner_cv=self.s_split,
                              verbosity=0,
                              project_folder='./tmp/')

    def simple_classification(self):
        dataset = load_breast_cancer()
        self.X = dataset["data"]
        self.y = dataset["target"]
        return self.X, self.y

    def test_photon_implementation_simple(self):
        """Integration test for simple pipeline without Switch."""
        # PHOTONAI implementation
        self.pipe.add(PipelineElement('StandardScaler'))
        self.pipe += PipelineElement('PCA', hyperparameters={'n_components': IntegerRange(5, 30)})
        self.pipe += PipelineElement('SVC', hyperparameters={'kernel': Categorical(["rbf", 'poly']),
                                                             'C': FloatRange(0.001, 2, range_type='logspace')},
                                     gamma='auto')
        self.X, self.y = self.simple_classification()
        self.pipe.fit(self.X, self.y)

        # direct AUTO ML implementation
        # Build Configuration Space which defines all parameters and their ranges
        cs = ng.p.Instrumentation(
            PCA__n_components=ng.p.Scalar(lower=5, upper=30).set_integer_casting(),
            SVC__kernel=ng.p.Choice(["rbf", 'poly']),
            SVC__C=ng.p.Log(lower=0.001, upper=2)
        )
        cs.random_state.seed(42)

        # Optimize, using a Nevergrad directly
        optimizer = ng.optimizers.NGO(parametrization=cs, budget=20)
        optimizer.minimize(self.objective_function_simple)

        runhistory_photon = [x.metrics_test[0].value for x in self.pipe.results.outer_folds[0].tested_config_list]
        runhistory_original = [1-x.mean for x in list(optimizer.optim.archive.bytesdict.values())]

        self.assertListEqual(runhistory_photon, runhistory_original)

    def objective_function_simple(self, **cfg):
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

    def test_facade(self):
        facades = list(ng.optimizers.registry.keys())

        with self.assertRaises(ValueError):
            NevergradOptimizer(facade="Nevergrad4BOO", n_configurations=10)

        for facade in facades:
            NevergradOptimizer(facade=facade, n_configurations=10)

    def test_other(self):
        opt = NevergradOptimizer(facade="NGO", n_configurations=10)
        pipeline_elements = [PipelineElement('SVC', hyperparameters={'kernel': ["sigmoid", "rbf"],
                                                                     'C': [0.6], 'coef0': Categorical([0.5])})]
        of = lambda x: x ** 2
        with warnings.catch_warnings(record=True) as w:
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)
            assert any("PHOTONAI has detected some" in s for s in [e.message.args[0] for e in w])

        pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': FloatRange(0.1, 0.5,
                                                                                     range_type='geomspace')})]
        opt = NevergradOptimizer(facade="NGO")
        with self.assertRaises(NotImplementedError):
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)

        pipeline_elements = [PipelineElement("SVC", hyperparameters={'C': NumberRange(1, 3, range_type='range')})]
        opt = NevergradOptimizer(facade="NGO")
        with self.assertRaises(ValueError):
            opt.prepare(pipeline_elements=pipeline_elements, maximize_metric=True, objective_function=of)
