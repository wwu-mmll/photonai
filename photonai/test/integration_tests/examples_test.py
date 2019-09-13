
import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_BUG_3_multi_switch(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/BUG_3_multi_switch.py").read())
        
    def test_anomaly_detector_wrapper_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/anomaly_detector_wrapper_example.py").read())
        
    def test_BUG_6_basic_branchSwitch(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/BUG_6_basic_branchSwitch.py").read())
        
    def test_MAYBEBUG_11_basic_branch_sourceStack_estimatorStack_withSwitch_withCovs_samplePairing(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/MAYBEBUG_11_basic_branch_sourceStack_estimatorStack_withSwitch_withCovs_samplePairing.py").read())
        
    def test_autoSklearn_wrapper_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/autoSklearn_wrapper_example.py").read())
        
    def test_ermutation_test(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/permutation_test.py").read())
        
    def test_group_split(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/group_split.py").read())
        
    def test_imbalanced_data(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/imbalanced_data.py").read())
        
    def test_sample_pairing_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/sample_pairing_example.py").read())
        
    def test_confounder_removal_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/confounder_removal_example.py").read())
        
    def test_skopt_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/optimizers/skopt_example.py").read())
        
    def test_mongodb(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/results/mongodb.py").read())
        
    def test_results_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/results/results_example.py").read())
        
    def test_ensemble_stack(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/ensemble_stack.py").read())
        
    def test_data_integration(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/data_integration.py").read())
        
    def test_ipeline_switch(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/pipeline_switch.py").read())
        
    def test_ipeline_stacking(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/pipeline_stacking.py").read())
        
    def test_regression_pipeline(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/regression_pipeline.py").read())
        
    def test_batching_elements(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/batching_elements.py").read())
        
    def test_ipeline_branches(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/pipeline_branches.py").read())
        
    def test_no_outer_cv_default_pipe(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/no_outer_cv_default_pipe.py").read())
        
    def test_classification_pipeline(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/basic/classification_pipeline.py").read())
        
    def test_inverse_transform(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/inverse_transform.py").read())
        
    def test_custom_mask(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/custom_mask.py").read())
        
    def test_tim_neuro_example(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/tim_neuro_example.py").read())
        
    def test_brain_mask(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/brain_mask.py").read())
        
    def test_neuro_branch(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/neuro_branch.py").read())
        
    def test_atlas_mapping(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/atlas_mapping.py").read())
        
    def test_brain_atlas(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/neuro/brain_atlas.py").read())
        
    def test_range_restrictor(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/range_restrictor.py").read())
        
    def test_register_elements(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/register_elements.py").read())
        
    def test_callbacks(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/callbacks.py").read())
        
    def test_custom_transformer(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/custom_elements/custom_transformer.py").read())
        
    def test_custom_estimator(self):
        exec(open("/home/nwinter/PycharmProjects/photon_projects/photon_core/photonai/test/integration_tests/../../examples/advanced/custom_elements/custom_estimator.py").read())
        