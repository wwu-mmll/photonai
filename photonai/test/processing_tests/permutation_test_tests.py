import uuid
from bson.objectid import ObjectId
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, OutputSettings, PipelineElement
from photonai.processing.permutation_test import PermutationTest
from photonai.processing.results_handler import ResultsHandler
from photonai.test.photon_base_test import PhotonBaseTest


class PermutationTestTests(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        super(PermutationTestTests, cls).setUpClass()

        cls.perm_id = uuid.uuid4()
        cls.wizard_obj_id = ObjectId()
        cls.hyperpipe = Hyperpipe("permutation_test_pipe",
                                   inner_cv = KFold(n_splits=4),
                                   outer_cv = KFold(n_splits=3),
                                   metrics=["accuracy", "balanced_accuracy"],
                                   best_config_metric="balanced_accuracy",
                                   output_settings=OutputSettings(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_results",
                                                                  wizard_object_id=str(cls.wizard_obj_id)),
                                   permutation_id=str(cls.perm_id) + "_reference")
        cls.hyperpipe += PipelineElement("StandardScaler")
        cls.hyperpipe += PipelineElement("SVC")
        cls.X, cls.y = load_breast_cancer(return_X_y=True)
        cls.hyperpipe.fit(cls.X, cls.y)

    def test_wizard_preparation(self):
        result = PermutationTest.prepare_for_wizard(str(self.perm_id), self.wizard_obj_id)
        returned_duration = result["estimated_duration"]
        cached_duration = self.hyperpipe.results.computation_end_time - self.hyperpipe.results.computation_start_time
        self.assertAlmostEqual(round(returned_duration.total_seconds(), 2), round(cached_duration.total_seconds(), 2), 2)
        # Todo: setup case where it is false
        self.assertTrue(result["usability"])

        my_handler = ResultsHandler()
        my_handler.load_from_mongodb(self.hyperpipe.output_settings.mongodb_connect_url, str(self.wizard_obj_id))
        self.assertEqual(my_handler.results.permutation_id, PermutationTest.get_mother_permutation_id(str(self.perm_id)))

    def test_find_reference(self):
        obj_id = self.hyperpipe.results._id
        returned_obj_id = PermutationTest.find_reference(self.hyperpipe.output_settings.mongodb_connect_url,
                                                         str(self.perm_id), False)._id
        self.assertEqual(obj_id, returned_obj_id)

        wizard_obj_id = str(self.wizard_obj_id)
        latest_item = PermutationTest.find_reference(self.hyperpipe.output_settings.mongodb_connect_url,
                                                     ObjectId(wizard_obj_id), True)
        self.assertEqual(latest_item.name, wizard_obj_id)






