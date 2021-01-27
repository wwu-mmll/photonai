#
#
# import warnings
# from os.path import join, dirname, realpath
# from pathlib import Path
# from photonai.helper.photon_base_test import PhotonBaseTest
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
#
#
# class TestRunExamples(PhotonBaseTest):
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.file = __file__
#         super(TestRunExamples, cls).setUpClass()
#
#     def setUp(self):
#         self.examples_folder = Path(dirname(realpath(__file__))).parent.parent.joinpath('examples')
#
#     def test_keras_dnn_regression(self):
#         exec(open(join(self.examples_folder, "neural_networks/keras_dnn_regression.py")).read(), locals(), globals())
#
#     def test_multi_layer_perceptron_classifier(self):
#         exec(open(join(self.examples_folder, "neural_networks/multi_layer_perceptron_classifier.py")).read(), locals(), globals())
#
#     def test_keras_dnn_callbacks(self):
#         exec(open(join(self.examples_folder, "neural_networks/keras_dnn_callbacks.py")).read(), locals(), globals())
#
#     def test_keras_cnn_classification(self):
#         exec(open(join(self.examples_folder, "neural_networks/keras_cnn_classification.py")).read(), locals(), globals())
#
#     def test_keras_dnn_multiclass_classification(self):
#         exec(open(join(self.examples_folder, "neural_networks/keras_dnn_multiclass_classification.py")).read(), locals(), globals())
#
#     def test_regression_with_constraints(self):
#         exec(open(join(self.examples_folder, "advanced/regression_with_constraints.py")).read(), locals(), globals())
#
#     def test_multiclass_classification(self):
#         exec(open(join(self.examples_folder, "advanced/multiclass_classification.py")).read(), locals(), globals())
#
#     def test_sample_pairing_example(self):
#         exec(open(join(self.examples_folder, "advanced/sample_pairing_example.py")).read(), locals(), globals())
#
#     def test_imbalanced_data(self):
#         exec(open(join(self.examples_folder, "advanced/imbalanced_data.py")).read(), locals(), globals())
#
#     def test_confounder_removal_example(self):
#         exec(open(join(self.examples_folder, "advanced/confounder_removal_example.py")).read(), locals(), globals())
#
#     def test_group_split(self):
#         exec(open(join(self.examples_folder, "advanced/group_split.py")).read(), locals(), globals())
#
#     def test_svc_ensemble(self):
#         exec(open(join(self.examples_folder, "advanced/svc_ensemble.py")).read(), locals(), globals())
#
#     def test_estimator_stack_voting(self):
#         exec(open(join(self.examples_folder, "advanced/estimator_stack_voting.py")).read(), locals(), globals())
#
#     def test_callbacks(self):
#         exec(open(join(self.examples_folder, "advanced/callbacks.py")).read(), locals(), globals())
#
#     def test_additional_data(self):
#         exec(open(join(self.examples_folder, "advanced/additional_data.py")).read(), locals(), globals())
#
#     def test_permutation_test(self):
#         exec(open(join(self.examples_folder, "advanced/permutation_test.py")).read(), locals(), globals())
#
#     def test_register_elements(self):
#         exec(open(join(self.examples_folder, "advanced/register_elements.py")).read(), locals(), globals())
#
#     def test_feature_selection(self):
#         exec(open(join(self.examples_folder, "advanced/feature_selection.py")).read(), locals(), globals())
#
#     def test_custom_estimator(self):
#         exec(open(join(self.examples_folder, "advanced/custom_elements/custom_estimator.py")).read(), locals(), globals())
#
#     def test_custom_transformer(self):
#         exec(open(join(self.examples_folder, "advanced/custom_elements/custom_transformer.py")).read(), locals(), globals())
#
#     def test_regression(self):
#         exec(open(join(self.examples_folder, "getting_started/regression.py")).read(), locals(), globals())
#
#     def test_no_outer_cv_default_pipe(self):
#         exec(open(join(self.examples_folder, "getting_started/no_outer_cv_default_pipe.py")).read(), locals(), globals())
#
#     def test_load_hyperpipe_from_json(self):
#         exec(open(join(self.examples_folder, "getting_started/load_hyperpipe_from_json.py")).read(), locals(), globals())
#
#     def test_batching_elements(self):
#         exec(open(join(self.examples_folder, "getting_started/batching_elements.py")).read(), locals(), globals())
#
#     def test_pipeline_branches(self):
#         exec(open(join(self.examples_folder, "getting_started/pipeline_branches.py")).read(), locals(), globals())
#
#     def test_stack(self):
#         exec(open(join(self.examples_folder, "getting_started/stack.py")).read(), locals(), globals())
#
#     def test_classification(self):
#         exec(open(join(self.examples_folder, "getting_started/classification.py")).read(), locals(), globals())
#
#     def test_data_integration(self):
#         exec(open(join(self.examples_folder, "getting_started/data_integration.py")).read(), locals(), globals())
#
#     def test_jmlr_example(self):
#         exec(open(join(self.examples_folder, "getting_started/jmlr_example.py")).read(), locals(), globals())
#
#     def test_switch(self):
#         exec(open(join(self.examples_folder, "getting_started/switch.py")).read(), locals(), globals())
#
#     def test_classifier_ensemble(self):
#         exec(open(join(self.examples_folder, "getting_started/classifier_ensemble.py")).read(), locals(), globals())
#
#     def test_skopt_example(self):
#         exec(open(join(self.examples_folder, "optimizer/skopt_example.py")).read(), locals(), globals())
#
#     def test_comparisson_example(self):
#         exec(open(join(self.examples_folder, "optimizer/comparisson_example.py")).read(), locals(), globals())
#
#     def test_smac_example(self):
#         exec(open(join(self.examples_folder, "optimizer/smac_example.py")).read(), locals(), globals())
#
#     def test_results_example(self):
#         exec(open(join(self.examples_folder, "results/results_example.py")).read(), locals(), globals())
#
#     def test_mongodb(self):
#         exec(open(join(self.examples_folder, "results/mongodb.py")).read(), locals(), globals())
