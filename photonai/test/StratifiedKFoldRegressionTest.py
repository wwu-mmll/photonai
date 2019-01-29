import unittest
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from sklearn.datasets import load_breast_cancer
from photonai.validation.cross_validation import StratifiedKFoldRegression
import numpy as np



class StratifiedRegressionTest(unittest.TestCase):

    def setUp(self):
        # set up all we need
        self.X, self.y = load_breast_cancer(True)
        self.groups = np.random.randint(-10, 10, (self.y.shape[0]))
        self.outer_cv = StratifiedKFoldRegression(n_splits=2, random_state=15, shuffle=True)
        self.inner_cv = StratifiedKFoldRegression(n_splits=2, random_state=15, shuffle=True)
        output = OutputSettings(save_output=False, save_predictions='None', save_feature_importances='None')

        # without group variable
        self.pipe = Hyperpipe('stratified_regression',
                              outer_cv=self.outer_cv,
                              inner_cv=self.inner_cv,
                              metrics=['mean_squared_error'],
                              best_config_metric='mean_squared_error',
                              output_settings=output)
        self.pipe += PipelineElement('SVC')
        self.pipe.X = self.X
        self.pipe.y = self.y
        self.pipe._generate_outer_cv_indices()

        # with group variable
        self.pipe_w_gr = Hyperpipe('stratified_regression',
                              outer_cv=self.outer_cv,
                              inner_cv=self.inner_cv,
                              metrics=['mean_squared_error'],
                              best_config_metric='mean_squared_error',
                              groups=self.groups,
                              output_settings=output)
        self.pipe_w_gr += PipelineElement('SVC')
        self.pipe_w_gr.X = self.X
        self.pipe_w_gr.y = self.y
        self.pipe_w_gr._generate_outer_cv_indices()

    def test_stratified_regression(self):
        splits = list()
        splits_pipe = list()
        for train, test in self.outer_cv.split(self.X, self.y):
            splits.append(train)
            splits.append(test)

        for train, test in self.pipe.data_test_cases:
            splits_pipe.append(train)
            splits_pipe.append(test)

        for i in range(len(splits)):
            np.testing.assert_array_equal(splits[i], splits_pipe[i])

    def test_stratified_regression_w_groups(self):
        splits = list()
        splits_pipe = list()
        for train, test in self.outer_cv.split(self.X, self.groups):
            splits.append(train)
            splits.append(test)

        for train, test in self.pipe_w_gr.data_test_cases:
            splits_pipe.append(train)
            splits_pipe.append(test)

        for i in range(len(splits)):
            np.testing.assert_array_equal(splits[i], splits_pipe[i])
