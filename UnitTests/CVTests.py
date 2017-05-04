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

    def test_cv_local_search_true(self):
        self.outer_hyperpipe.fit(self.X, self.y)

        print('local_search true: outer pipeline data:')
        print(self.outer_pipeline_test_element.base_element.data_dict['fit_X'])

        print('local_search true: inner pipeline data:')
        print(self.inner_pipeline_test_element.base_element.data_dict['fit_X'])

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X']
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X']
        # with KFold = 2 we expect the
        self.assertListEqual(outer_data[0:25].tolist(), inner_data.tolist())

    def test_primary_train_test_split(self):

        self.outer_hyperpipe.test_size = 0.2
        self.outer_hyperpipe.fit(self.X, self.y)

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X']
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X']

        # test that it is only 80% of original X and that there is a test_x of 20% size
        self.assertEqual(outer_data.shape[0], 40)
        # test that outer data is 50% from 80% or original
        self.assertEqual(inner_data.shape[0], 20)


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

    def test_cv_local_search_false(self):
        self.outer_hyperpipe.fit(self.X, self.y)

        print('local_search true: outer pipeline data:')
        print(self.outer_pipeline_test_element.base_element.data_dict['fit_X'])

        print('local_search true: inner pipeline data:')
        print(self.inner_pipeline_test_element.base_element.data_dict['fit_X'])

        outer_data = self.outer_pipeline_test_element.base_element.data_dict['fit_X']
        inner_data = self.inner_pipeline_test_element.base_element.data_dict['fit_X']
        # with KFold = 2 we expect the
        self.assertListEqual(outer_data.tolist(), inner_data.tolist())


if __name__ == '__main__':
    unittest.main()
