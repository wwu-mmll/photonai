
import unittest
import warnings

from photonai.test.PhotonBaseTest import PhotonBaseTest

from os.path import join, isdir
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TestRunExamples(PhotonBaseTest):
    
    def setUp(self):
        self.examples_folder = "../examples"
        if not isdir(self.examples_folder):
            self.examples_folder = "../../examples"

    def test_brain_mask(self):
        exec(open(join(self.examples_folder, "neuro/brain_mask.py")).read(), locals(), globals())

    def test_custom_mask(self):
        exec(open(join(self.examples_folder, "neuro/custom_mask.py")).read(), locals(), globals())

    def test_oasis_age_prediction(self):
        exec(open(join(self.examples_folder, "neuro/oasis_age_prediction.py")).read(), locals(), globals())

    def test_neuro_branch(self):
        exec(open(join(self.examples_folder, "neuro/neuro_branch.py")).read(), locals(), globals())

    def test_atlas_mapping(self):
        exec(open(join(self.examples_folder, "neuro/atlas_mapping.py")).read(), locals(), globals())

    def test_inverse_transform(self):
        exec(open(join(self.examples_folder, "neuro/inverse_transform.py")).read(), locals(), globals())

    def test_brain_atlas(self):
        exec(open(join(self.examples_folder, "neuro/brain_atlas.py")).read(), locals(), globals())

    def test_multi_layer_perceptron_classifier(self):
        exec(open(join(self.examples_folder, "neural_networks/multi_layer_perceptron_classifier.py")).read(), locals(), globals())

    def test_range_restrictor(self):
        exec(open(join(self.examples_folder, "advanced/range_restrictor.py")).read(), locals(), globals())

    def test_sample_pairing_example(self):
        exec(open(join(self.examples_folder, "advanced/sample_pairing_example.py")).read(), locals(), globals())

    def test_imbalanced_data(self):
        exec(open(join(self.examples_folder, "advanced/imbalanced_data.py")).read(), locals(), globals())

    def test_confounder_removal_example(self):
        exec(open(join(self.examples_folder, "advanced/confounder_removal_example.py")).read(), locals(), globals())

    def test_group_split(self):
        exec(open(join(self.examples_folder, "advanced/group_split.py")).read(), locals(), globals())

    def test_callbacks(self):
        exec(open(join(self.examples_folder, "advanced/callbacks.py")).read(), locals(), globals())

    def test_permutation_test(self):
        exec(open(join(self.examples_folder, "advanced/permutation_test.py")).read(), locals(), globals())

    def test_register_elements(self):
        exec(open(join(self.examples_folder, "advanced/register_elements.py")).read(), locals(), globals())

    def test_custom_estimator(self):
        exec(open(join(self.examples_folder, "advanced/custom_elements/custom_estimator.py")).read(), locals(), globals())

    def test_custom_transformer(self):
        exec(open(join(self.examples_folder, "advanced/custom_elements/custom_transformer.py")).read(), locals(), globals())

    def test_regression(self):
        exec(open(join(self.examples_folder, "basic/regression.py")).read(), locals(), globals())

    def test_no_outer_cv_default_pipe(self):
        exec(open(join(self.examples_folder, "basic/no_outer_cv_default_pipe.py")).read(), locals(), globals())

    def test_batching_elements(self):
        exec(open(join(self.examples_folder, "basic/batching_elements.py")).read(), locals(), globals())

    def test_pipeline_branches(self):
        exec(open(join(self.examples_folder, "basic/pipeline_branches.py")).read(), locals(), globals())

    def test_stack(self):
        exec(open(join(self.examples_folder, "basic/stack.py")).read(), locals(), globals())

    def test_classification(self):
        exec(open(join(self.examples_folder, "basic/classification.py")).read(), locals(), globals())

    def test_data_integration(self):
        exec(open(join(self.examples_folder, "basic/data_integration.py")).read(), locals(), globals())

    def test_switch(self):
        exec(open(join(self.examples_folder, "basic/switch.py")).read(), locals(), globals())

    def test_classifier_ensemble(self):
        exec(open(join(self.examples_folder, "basic/classifier_ensemble.py")).read(), locals(), globals())

    def test_skopt_example(self):
        exec(open(join(self.examples_folder, "optimizer/skopt_example.py")).read(), locals(), globals())

    def test_results_example(self):
        exec(open(join(self.examples_folder, "results/results_example.py")).read(), locals(), globals())

    def test_mongodb(self):
        exec(open(join(self.examples_folder, "results/mongodb.py")).read(), locals(), globals())
