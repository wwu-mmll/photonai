import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_skopt_example(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/optimizer/skopt_example.py").read(),
             locals(), globals())
        
    def test_mongodb(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/results/mongodb.py").read(),
             locals(), globals())
        
    def test_results_example(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/results/results_example.py").read(),
             locals(), globals())
        
    def test_data_integration(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/data_integration.py").read(),
             locals(), globals())
        
    def test_classification(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/classification.py").read(),
             locals(), globals())
        
    def test_switch(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/switch.py").read(),
             locals(), globals())
        
    def test_stack(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/stack.py").read(),
             locals(), globals())
        
    def test_batching_elements(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/batching_elements.py").read(),
             locals(), globals())
        
    def test_regression(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/regression.py").read(),
             locals(), globals())
        
    def test_pipeline_branches(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/pipeline_branches.py").read(),
             locals(), globals())
        
    def test_no_outer_cv_default_pipe(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/no_outer_cv_default_pipe.py").read(),
             locals(), globals())
        
    def test_classifier_ensemble(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/classifier_ensemble.py").read(),
             locals(), globals())
        
    def test_inverse_transform(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/inverse_transform.py").read(),
             locals(), globals())
        
    def test_custom_mask(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/custom_mask.py").read(),
             locals(), globals())

    def test_oasis_age_prediction(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/oasis_age_prediction.py").read(),
             locals(), globals())
        
    def test_brain_mask(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/brain_mask.py").read(),
             locals(), globals())
        
    def test_neuro_branch(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/neuro_branch.py").read(),
             locals(), globals())
        
    def test_atlas_mapping(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/atlas_mapping.py").read(),
             locals(), globals())
        
    def test_brain_atlas(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/brain_atlas.py").read(),
             locals(), globals())
        
    def test_multi_layer_perceptron_classifier(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neural_networks/multi_layer_perceptron_classifier.py").read(),
             locals(), globals())
        
    def test_range_restrictor(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/range_restrictor.py").read(),
             locals(), globals())
        
    def test_register_elements(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/register_elements.py").read(),
             locals(), globals())
        
    def test_permutation_test(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/permutation_test.py").read(),
             locals(), globals())
        
    def test_group_split(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/group_split.py").read(),
             locals(), globals())
        
    def test_imbalanced_data(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/imbalanced_data.py").read(),
             locals(), globals())
        
    def test_sample_pairing_example(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/sample_pairing_example.py").read(),
             locals(), globals())
        
    def test_confounder_removal_example(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/confounder_removal_example.py").read(),
             locals(), globals())
        
    def test_callbacks(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/callbacks.py").read(),
             locals(), globals())
        
    def test_custom_transformer(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/custom_elements/custom_transformer.py").read(),
             locals(), globals())
        
    def test_custom_estimator(self):
        exec(open(
            "/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/custom_elements/custom_estimator.py").read(),
             locals(), globals())
