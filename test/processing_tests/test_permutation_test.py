import uuid
import numpy as np
from bson.objectid import ObjectId
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, OutputSettings, PipelineElement
from photonai.processing.permutation_test import PermutationTest
from photonai.processing.results_handler import ResultsHandler
from photonai.helper.photon_base_test import PhotonBaseTest


class PermutationTestTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(PermutationTestTests, cls).setUpClass()

        cls.perm_id = uuid.uuid4()
        cls.wizard_obj_id = ObjectId()
        cls.hyperpipe = Hyperpipe("permutation_test_pipe",
                                  inner_cv=KFold(n_splits=4),
                                  outer_cv=KFold(n_splits=3),
                                  metrics=["accuracy", "balanced_accuracy"],
                                  best_config_metric="balanced_accuracy",
                                  output_settings=OutputSettings(
                                      mongodb_connect_url="mongodb://localhost:27017/photon_results",
                                      wizard_object_id=str(cls.wizard_obj_id)),
                                  project_folder=cls.tmp_folder_path,
                                  verbosity=0,
                                  permutation_id=str(cls.perm_id) + "_reference")
        cls.hyperpipe += PipelineElement("StandardScaler")
        cls.hyperpipe += PipelineElement("SVC")
        cls.X, cls.y = load_breast_cancer(return_X_y=True)
        cls.hyperpipe.fit(cls.X, cls.y)

    def test_wizard_preparation(self):
        result = PermutationTest.prepare_for_wizard(str(self.perm_id), self.wizard_obj_id,
                                                    mongo_db_connect_url="mongodb://localhost:27017/photon_results")
        returned_duration = result["estimated_duration"]
        cached_duration = self.hyperpipe.results.computation_end_time - self.hyperpipe.results.computation_start_time
        self.assertAlmostEqual(round(returned_duration.total_seconds(), 3),
                               round(cached_duration.total_seconds(), 3), 2)
        # Todo: setup case where it is false
        self.assertTrue(result["usability"])

        my_handler = ResultsHandler()
        my_handler.load_from_mongodb(self.hyperpipe.output_settings.mongodb_connect_url, str(self.wizard_obj_id))
        self.assertEqual(my_handler.results.permutation_id,
                         PermutationTest.get_mother_permutation_id(str(self.perm_id)))

    def test_find_reference(self):
        obj_id = self.hyperpipe.results._id
        returned_obj_id = PermutationTest.find_reference(self.hyperpipe.output_settings.mongodb_connect_url,
                                                         str(self.perm_id), False)._id
        self.assertEqual(obj_id, returned_obj_id)

        wizard_obj_id = str(self.wizard_obj_id)
        latest_item = PermutationTest.find_reference(self.hyperpipe.output_settings.mongodb_connect_url,
                                                     ObjectId(wizard_obj_id), True)
        self.assertEqual(latest_item.name, wizard_obj_id)

    def create_hyperpipe(self):
        # this is needed here for the parallelisation
        from photonai.base import Hyperpipe, PipelineElement, OutputSettings
        from photonai.optimization import IntegerRange
        from sklearn.model_selection import GroupKFold
        from sklearn.model_selection import KFold

        settings = OutputSettings(mongodb_connect_url='mongodb://localhost:27017/photon_results')
        my_pipe = Hyperpipe('permutation_test_1',
                            optimizer='grid_search',
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='accuracy',
                            outer_cv=GroupKFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            calculate_metrics_across_folds=True,
                            use_test_set=True,
                            verbosity=0,
                            project_folder=self.tmp_folder_path,
                            output_settings=settings)

        # Add transformer elements
        my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                                   test_disabled=False, with_mean=True, with_std=True)

        my_pipe += PipelineElement("PCA", hyperparameters={'n_components': IntegerRange(3, 5)},
                                   test_disabled=False)

        # Add estimator
        my_pipe += PipelineElement("SVC", hyperparameters={'kernel': ['linear', 'rbf']},  # C': FloatRange(0.1, 5),
                                   gamma='scale', max_iter=1000000)

        return my_pipe

    def create_hyperpipe_no_mongo(self):
        from photonai.base import Hyperpipe
        from sklearn.model_selection import KFold

        my_pipe = Hyperpipe('permutation_test_1',
                            optimizer='grid_search',
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='accuracy',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            calculate_metrics_across_folds=True,
                            use_test_set=True,
                            verbosity=0,
                            project_folder=self.tmp_folder_path)
        return my_pipe

    def test_no_mongo_connection_string(self):
        perm_tester = PermutationTest(self.create_hyperpipe_no_mongo, n_perms=2, n_processes=3, random_state=11,
                                      permutation_id=str(uuid.uuid4()))
        with self.assertRaises(ValueError):
            perm_tester.fit(self.X, self.y)

    def test_run_parallelized_perm_test(self):
        X, y = load_breast_cancer(return_X_y=True)
        my_perm_id = str(uuid.uuid4())
        groups = np.random.random_integers(0, 3, (len(y),))
        perm_tester = PermutationTest(self.create_hyperpipe, n_perms=2, n_processes=3, random_state=11,
                                      permutation_id=my_perm_id)
        perm_tester.fit(X, y, groups=groups)

    def test_setup_non_useful_perm_test(self):
        np.random.seed(1335)
        X, y = np.random.random((200, 5)), np.random.randint(0, 2, size=(200, ))
        my_perm_id = str(uuid.uuid4())
        groups = np.random.random_integers(0, 3, (len(y),))
        perm_tester = PermutationTest(self.create_hyperpipe, n_perms=2, n_processes=3, random_state=11,
                                      permutation_id=my_perm_id)
        with self.assertRaises(RuntimeError):
            perm_tester.fit(X, y, groups=groups)

    def test_run_perm_test(self):
        X, y = load_breast_cancer(return_X_y=True)
        my_perm_id = str(uuid.uuid4())
        groups = np.random.random_integers(0, 3, (len(y),))
        perm_tester = PermutationTest(self.create_hyperpipe, n_perms=2, n_processes=1, random_state=11,
                                      permutation_id=my_perm_id)
        perm_tester.fit(X, y, groups=groups)

        results = PermutationTest._calculate_results(my_perm_id,
                                                     mongodb_path='mongodb://localhost:27017/photon_results')

        self.assertAlmostEqual(results.p_values['accuracy'], 0)
