import unittest
import types
import numpy as np
import warnings

from photonai.processing.metrics import Scorer, spearman_correlation, specificity, sensitivity, one_hot_to_binary, \
    pearson_correlation, balanced_accuracy, categorical_accuracy_score, variance_explained_score


class ScorerTest(unittest.TestCase):

    def setUp(self):
        """
        Set up for Scorer Tests.
        """
        self.all_implemented_metrics = Scorer.ELEMENT_DICTIONARY.keys()
        self.some_not_implemented_metrics = ["abc_metric", "photon_metric"]

    def test_create(self):
        """
        Test for method create.
        Should searching for the metric by name and instantiates the according calculation function
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIsInstance(Scorer.create(implemented_metric), types.FunctionType)

        for not_implemented_metric in self.some_not_implemented_metrics:
            with self.assertRaises(NameError):
                self.assertIsNone(Scorer.create(not_implemented_metric))

    def test_greater_is_better_distinction(self):
        """
        Test for method greater_is_better_distinction.
        Should return Boolean or raise NotImplementedError.
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIn(Scorer.greater_is_better_distinction(implemented_metric), [True, False])

        for not_implemented_metric in self.some_not_implemented_metrics:
            with self.assertRaises(NameError):
                Scorer.greater_is_better_distinction(not_implemented_metric)

    def test_calculate_metrics(self):
        """
        Test for method calculate_metrics.
        Handle all given metrics with a scorer call.
        """
        for implemented_metric in self.all_implemented_metrics:
            self.assertIsInstance(Scorer.calculate_metrics([1, 1, 0, 1],
                                                           [0, 1, 0, 1],
                                                           [implemented_metric])[implemented_metric], float)

        for not_implemented_metric in self.some_not_implemented_metrics:
            with self.assertRaises(NameError):
                np.testing.assert_equal(Scorer.calculate_metrics(
                    [1, 1, 0, 1], [0, 1, 0, 1], [not_implemented_metric])[not_implemented_metric], np.nan)

    def test_doubled_custom_metric(self):

        def custom_metric(y_true, y_pred):
            return 99.9

        Scorer.register_custom_metric(('a_custom_metric', custom_metric))

        with self.assertRaises(ValueError):
            Scorer.register_custom_metric(None)

        with warnings.catch_warnings(record=True) as w:
            Scorer.register_custom_metric(('a_custom_metric', custom_metric))
            assert any("is ambiguous. Please specify metric" in s for s in [e.message.args[0] for e in w])

    def test_keras_metric(self):
        try:
            from keras.metrics import MeanAbsoluteError
            Scorer.register_custom_metric(MeanAbsoluteError)

            Scorer.register_custom_metric('any_weird_metric')
        except ImportError:
            pass

    def test_photonai_metrics(self):
        y_true = np.concatenate((np.ones((200,)), np.zeros((400,))))
        y_pred = np.concatenate((np.ones((100,)), np.zeros((500,))))

        pearson_corr = pearson_correlation(y_true, y_pred)
        self.assertAlmostEqual(pearson_corr, 0.6324555320336789)
        spearman_corr = spearman_correlation(y_true, y_pred)
        self.assertAlmostEqual(spearman_corr, 0.632455532033676)
        cat_acc = categorical_accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(cat_acc, 0.8333333333333334)
        v_explained = variance_explained_score(y_true, y_pred)
        self.assertAlmostEqual(v_explained, 0.4000000000000038)

        sens = sensitivity(y_true, y_pred)
        self.assertEqual(sens, 0.5)
        spec = specificity(y_true, y_pred)
        self.assertEqual(spec, 1.0)
        b_acc = balanced_accuracy(y_true, y_pred)
        self.assertEqual(b_acc, 0.75)

        # get np.nan for multidim
        y_multiclass = np.concatenate((np.ones((200,)), np.zeros((400,)),
                                       np.ones(100,)*3))

        self.assertTrue(np.isnan(sensitivity(y_multiclass, y_multiclass)))
        self.assertTrue(np.isnan(specificity(y_multiclass, y_multiclass)))
        self.assertTrue(np.isnan(balanced_accuracy(y_multiclass, y_multiclass)))

    def test_one_hot_decoding(self):
        y_one_hot = np.stack((np.concatenate((np.ones((100,)), np.zeros((100,)))),
                              np.concatenate((np.zeros((100,)), np.ones((100,))))), axis=1)
        binarized_multidim = one_hot_to_binary(y_one_hot)
        self.assertTrue(np.array_equal(binarized_multidim, np.concatenate((np.zeros((100,)), np.ones((100, ))))))
