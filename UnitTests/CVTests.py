import unittest
import numpy as np
from sklearn.model_selection import KFold
from HPOFramework.HPOBaseClasses import PipelineElement, Hyperpipe, PipelineFusion


class CVTestsLocalSearchTrue(unittest.TestCase):

    def setUp(self):
        # set up inner pipeline
        self.inner_hyperpipe = Hyperpipe('inner_pipe', KFold(n_splits=2), local_search=True)
        self.inner_pipeline_test_element = PipelineElement.create('test_wrapper')
        self.inner_hyperpipe += self.inner_pipeline_test_element
        self.pipeline_fusion = PipelineFusion('fusion_element', [self.inner_hyperpipe])
        # set up outer pipeline
        self.outer_hyperpipe = Hyperpipe('outer_pipe', KFold(n_splits=2))
        self.outer_pipeline_test_element = PipelineElement.create('test_wrapper')
        self.outer_hyperpipe += self.outer_pipeline_test_element
        self.outer_hyperpipe += self.pipeline_fusion

        self.X = np.arange(1, 101)
        self.y = np.ones((100,))

    def test_default_split_fit(self):
        """
        test default splitting mode: 80% validation and 20% testing
        make sure that DURING the optimization the optimum pipeline is fitted with the correct amount of data
        """
        self.outer_hyperpipe.debug_cv_mode = True
        self.inner_hyperpipe.debug_cv_mode = True

        self.outer_hyperpipe.fit(self.X, self.y)

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        print('local_search true: outer pipeline data:')
        print(sorted(outer_data))

        print('local_search true: inner pipeline data:')
        print(sorted(inner_data))

        # we expect that all items from inner_data are existent in outer_data
        validation = set(inner_data) < set(outer_data)
        self.assertTrue(validation)
        # test that it is only 50% of 80% of original X (n=100) and that there is a test_x of 20% size
        self.assertEqual(len(outer_data), 40)
        # test that inner data is 50% from 80% of outer
        self.assertEqual(len(inner_data), 16)
        # we also expect that inner_data is 50% of length from outer_data
        self.assertEqual(len(inner_data), 0.5*0.8*len(outer_data))


    def test_default_split_predict(self):
        """
        test default splitting mode: 80% validation and 20% testing
        make sure that AFTER the optimization the optimum pipeline is fitted with the correct amount of data
        which means test that the optimum pipe is fitted to the validation data and tested with the test data
        """
        self.outer_hyperpipe.debug_cv_mode = False
        self.inner_hyperpipe.debug_cv_mode = False

        self.outer_hyperpipe.fit(self.X, self.y)

        print('local_search true: outer pipeline data:')
        print(self.outer_pipeline_test_element.base_element.data_dict['fit_X'])

        print('local_search true: inner pipeline data:')
        print(self.inner_pipeline_test_element.base_element.data_dict['fit_X'])

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        # we expect that all items from inner_data are existent in outer_data
        validation = set(inner_data) < set(outer_data)
        self.assertTrue(validation)
        # test that it is only 80% of original X (n=100) and that there is a test_x of 20% size
        self.assertEqual(len(outer_data), 80)
        # test that inner data is 80% from 80% of original
        self.assertEqual(len(inner_data), 64)
        # we also expect that inner_data is 80% of length from outer_data
        self.assertEqual(len(inner_data), 0.8*len(outer_data))

    def test_no_split(self):
        """
        test no splitting mode: the data is NOT split into test and validation set
        """
        self.outer_hyperpipe.debug_cv_mode = True
        self.outer_hyperpipe.eval_final_performance = False
        self.inner_hyperpipe.debug_cv_mode = True
        self.inner_hyperpipe.eval_final_performance = False

        self.outer_hyperpipe.fit(self.X, self.y)
        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        # we expect that all items from inner_data are existent in outer_data
        validation = set(inner_data) < set(outer_data)
        self.assertTrue(validation)
        # test that it is only 50% of original X (n=100)
        self.assertEqual(len(outer_data), 50)
        # test that inner data is 50% from 50% of outer = 25% of original
        self.assertEqual(len(inner_data), 25)

    def test_CV_split(self):
        """
        test cv splitting mode: the entire search for hyperparameters is cross validated
        """
        self.outer_hyperpipe.debug_cv_mode = True
        self.outer_hyperpipe.eval_final_performance = False
        self.outer_hyperpipe.hyperparameter_fitting_cv_object = KFold(n_splits=2)
        self.inner_hyperpipe.debug_cv_mode = True
        self.inner_hyperpipe.eval_final_performance = False
        self.inner_hyperpipe.hyperparameter_fitting_cv_object = KFold(n_splits=2)

        self.outer_hyperpipe.fit(self.X, self.y)
        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        # we expect that all items from inner_data are existent in outer_data
        validation = set(inner_data) < set(outer_data)
        self.assertTrue(validation)
        # we use KFold = 2 so the original should be 100/2 = 50
        # test that it is only 50% of original X (n=50)
        self.assertEqual(len(outer_data), 25)
        # test that inner data is 25% of outer = 12,5% of original
        self.assertTrue((len(inner_data) == 6 or len(inner_data) == 7))


class CVTestsLocalSearchFalse(unittest.TestCase):

    def setUp(self):
        self.outer_hyperpipe = Hyperpipe('outer_pipe', KFold(n_splits=2))

        # set up inner pipeline
        self.inner_hyperpipe = Hyperpipe('inner_pipe', KFold(n_splits=2), optimizer=self.outer_hyperpipe.optimizer,
                                         local_search=False)
        self.inner_pipeline_test_element = PipelineElement.create('test_wrapper')
        self.inner_hyperpipe += self.inner_pipeline_test_element
        self.pipeline_fusion = PipelineFusion('fusion_element', [self.inner_hyperpipe])

        # set up outer pipeline
        self.outer_pipeline_test_element = PipelineElement.create('test_wrapper')
        self.outer_hyperpipe += self.outer_pipeline_test_element
        self.outer_hyperpipe += self.pipeline_fusion

        self.X = np.arange(1, 101)
        self.y = np.ones((100,))

        self.inner_hyperpipe.debug_cv_mode = True
        self.outer_hyperpipe.debug_cv_mode = True

    def test_no_split(self):

        self.outer_hyperpipe.eval_final_performance = False
        self.inner_hyperpipe.eval_final_performance = False

        self.outer_hyperpipe.fit(self.X, self.y)

        print('local_search true: outer pipeline data:')
        print(self.outer_pipeline_test_element.base_element.data_dict['fit_X'])

        print('local_search true: inner pipeline data:')
        print(self.inner_pipeline_test_element.base_element.data_dict['fit_X'])

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        self.assertTrue(set(outer_data) == set(inner_data))
        self.assertEqual(len(outer_data), 50)

    def test_default_split(self):

        self.outer_hyperpipe.eval_final_performance = True
        self.inner_hyperpipe.eval_final_performance = True

        self.outer_hyperpipe.fit(self.X, self.y)

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        self.assertTrue(set(outer_data) == set(inner_data))
        self.assertEqual(len(outer_data), 40)

    def test_cv_split(self):

        self.outer_hyperpipe.hyperparameter_fitting_cv_object = KFold(n_splits=2)
        # should be ignored:
        self.inner_hyperpipe.hyperparameter_fitting_cv_object = KFold(n_splits=2)

        self.outer_hyperpipe.fit(self.X, self.y)

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X'].tolist()
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X'].tolist()

        self.assertTrue(set(outer_data) == set(inner_data))
        self.assertEqual(len(outer_data), 25)
        self.assertEqual(len(outer_data), len(inner_data))

if __name__ == '__main__':
    unittest.main()
