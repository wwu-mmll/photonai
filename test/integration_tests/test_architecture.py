from itertools import product

import numpy as np
from sklearn.datasets import make_regression, make_classification, load_iris
from sklearn.model_selection import KFold, ShuffleSplit, LeaveOneOut

from photonai.base import Hyperpipe, PipelineElement, Switch, Stack, Branch, DataFilter
from photonai.optimization import Categorical, FloatRange, IntegerRange
from photonai.helper.photon_base_test import PhotonBaseTest


class TestArchitectures(PhotonBaseTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = __file__
        super(TestArchitectures, cls).setUpClass()
        n_samples = 40

        cls.test_multiple_hyperpipes = False
        cls.hyperpipes = list()
        cls.regression = 'regression'
        cls.classification = 'classification'

        if cls.test_multiple_hyperpipes:
            optimizer_list = ['random_grid_search', 'sk_opt']
            eval_final_performance_list = [True, False]
            inner_cv_list = [KFold(n_splits=3, shuffle=True), ShuffleSplit(n_splits=1, test_size=.2), LeaveOneOut()]
            outer_cv_list = [None, KFold(n_splits=3, shuffle=True), ShuffleSplit(n_splits=1, test_size=.25),
                             LeaveOneOut()]
            performance_constraints_list = [None]

            combinations = list(product(optimizer_list, eval_final_performance_list, inner_cv_list, outer_cv_list,
                                performance_constraints_list))
            for optimizer, eval_final_performance, inner_cv, outer_cv, performance_constraints in combinations:
                cls.hyperpipes.append(cls.create_hyperpipes(optimizer=optimizer,
                                                            inner_cv=inner_cv,
                                                            outer_cv=outer_cv,
                                                            eval_final_performance=eval_final_performance,
                                                            performance_constraints=performance_constraints,
                                                            cache_folder=cls.cache_folder_path,
                                                            tmp_folder=cls.tmp_folder_path))
        else:
            cls.hyperpipes.append(cls.create_hyperpipes())

        cls.regression_data = make_regression(n_samples=n_samples, n_features=20)
        cls.classification_data = make_classification(n_samples=n_samples, n_features=20)
        cls.X_shape = cls.regression_data[0].shape
        cls.groups = np.random.randint(low=1, high=3 + 1, size=n_samples)
        cls.cov1 = np.random.rand(n_samples)
        cls.cov2 = np.random.rand(n_samples)

    @staticmethod
    def create_hyperpipes(metrics: list = None, inner_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                          outer_cv=ShuffleSplit(n_splits=1, test_size=.2),
                          optimizer: str = 'random_grid_search',
                          optimizer_params: dict = {'n_configurations': 10}, eval_final_performance: bool = True,
                          performance_constraints: list = None, cache_folder='./cache', tmp_folder='./tmp'):

        pipe = Hyperpipe(name="architecture_test_pipe",
                         project_folder=tmp_folder,
                         optimizer=optimizer,
                         optimizer_params=optimizer_params,
                         best_config_metric='accuracy',
                         metrics=metrics,
                         inner_cv=inner_cv,
                         outer_cv=outer_cv,
                         use_test_set=eval_final_performance,
                         performance_constraints=performance_constraints,
                         cache_folder=cache_folder,
                         verbosity=0)
        return pipe

    @staticmethod
    def add_metrics(pipe, analysis_type):
        if analysis_type == 'classification':
            pipe.optimization.metrics = ['balanced_accuracy', 'accuracy', 'f1_score']
            pipe.optimization.best_config_metric = 'balanced_accuracy'
            return pipe
        elif analysis_type == 'regression':
            pipe.optimization.metrics = ['mean_absolute_error', 'mean_squared_error', 'pearson_correlation']
            pipe.optimization.best_config_metric = 'mean_squared_error'
            return pipe
        else:
            raise NotImplementedError("Only regression and classification is supported at the moment.")

    def run_hyperpipe(self, pipe, analysis_type):
        pipe = self.add_metrics(pipe, analysis_type)
        if analysis_type == 'classification':
            pipe.fit(self.classification_data[0], self.classification_data[1], **{'groups': self.groups,
                                                                                  'cov1': self.cov1,
                                                                                  'cov2': self.cov2})

        elif analysis_type == 'regression':
            pipe.fit(self.regression_data[0], self.regression_data[1], **{'groups': self.groups,
                                                                          'cov1': self.cov1,
                                                                          'cov2': self.cov2})

    def test_regression_1(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()
            pipe += PipelineElement(name='SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf'])})

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_2(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Switch
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor',
                                      hyperparameters={'min_samples_split': FloatRange(start=.05,
                                                                                       step=.1,
                                                                                       stop=.26,
                                                                                       range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_3(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # estimator Switch without hyperparameters
            my_switch = Switch('estimator_switch')
            my_switch += PipelineElement('SVR')
            my_switch += PipelineElement('RandomForestRegressor')
            pipe += my_switch

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_4(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Transformer Switch
            my_switch = Switch('trans_switch')
            my_switch += PipelineElement('PCA')
            my_switch += PipelineElement('FRegressionSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += my_switch
            pipe += PipelineElement('RandomForestRegressor')

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_5(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # multi-switch
            # setup switch to choose between PCA or simple feature selection and add it to the pipe
            pre_switch = Switch('preproc_switch')
            pre_switch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                          test_disabled=True)
            pre_switch += PipelineElement('FRegressionSelectPercentile',
                                          hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')},
                                          test_disabled=True)
            pipe += pre_switch
            # setup estimator switch and add it to the pipe
            estimator_switch = Switch('estimator_switch')
            estimator_switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                                        'C': Categorical([.01, 1, 5])})
            estimator_switch += PipelineElement('RandomForestRegressor',
                                                hyperparameters={'min_samples_split':
                                                                     FloatRange(start=.05, step=.1, stop=.26, range_type='range')})

            pipe += estimator_switch
            self.run_hyperpipe(pipe, self.regression)

    def test_regression_6(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Stack (use mean in the end)
            SVR = PipelineElement('SVR',
                                  hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestRegressor',
                                 hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += Stack('estimator_stack', elements=[SVR, RF])
            pipe += PipelineElement('PhotonVotingRegressor')

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_7(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Stack, but use same machine twice
            SVR1 = PipelineElement('SVR',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVR2 = PipelineElement('SVR',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            pipe += Stack('estimator_stack', elements=[SVR1, SVR2])
            pipe += PipelineElement('PhotonVotingRegressor')

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_8(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            pipe += PipelineElement('StandardScaler')
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_9(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # sample pairing with confounder removal
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
            pipe += PipelineElement('SamplePairingRegression',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=False)
            pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                            'C': Categorical([.01, 1, 5])})

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_10(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingRegression',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)
            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])
            # final estimator with stack output as features
            pipe += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})

            self.run_hyperpipe(pipe, self.regression)

    def test_regression_11(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingRegression',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.regression)

    def test_classification_1(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            pipe += PipelineElement(name='SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf'])})

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_2(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Switch
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_3(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # estimator Switch without hyperparameters
            my_switch = Switch('estimator_switch')
            my_switch += PipelineElement('SVC')
            my_switch += PipelineElement('RandomForestClassifier')
            pipe += my_switch

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_4(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Transformer Switch
            my_switch = Switch('trans_switch')
            my_switch += PipelineElement('PCA')
            my_switch += PipelineElement('FClassifSelectPercentile', hyperparameters={
                'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += my_switch
            pipe += PipelineElement('RandomForestClassifier')

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_5(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # multi-switch
            # setup switch to choose between PCA or simple feature selection and add it to the pipe
            pre_switch = Switch('preproc_switch')
            pre_switch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                          test_disabled=True)
            pre_switch += PipelineElement('FClassifSelectPercentile', hyperparameters={
                'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += pre_switch
            # setup estimator switch and add it to the pipe
            estimator_switch = Switch('estimator_switch')
            estimator_switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                                        'C': Categorical([.01, 1, 5])})
            estimator_switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += estimator_switch

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_6(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Stack (use mean in the end)
            SVR = PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                          'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += Stack('estimator_stack', elements=[SVR, RF])
            pipe += PipelineElement('PhotonVotingClassifier')

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_7(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # Simple estimator Stack, but use same machine twice
            SVC1 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVC2 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            pipe += Stack('estimator_stack', elements=[SVC1, SVC2])
            pipe += PipelineElement('PhotonVotingClassifier')

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_8(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            pipe += PipelineElement('StandardScaler')
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_9(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingClassification',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)
            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])
            # final estimator with stack output as features
            pipe += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_10(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()

            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingClassification',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(self.X_shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(self.X_shape[1] / 2)), stop=self.X_shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_11(self):
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()
            # Simple estimator Stack (train Random Forest on estimator stack proba outputs)
            # create estimator stack
            SVC1 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVC2 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestClassifier')
            # add to pipe
            pipe += Stack('estimator_stack', elements=[SVC1, SVC2, RF], use_probabilities=True)
            pipe += PipelineElement('RandomForestClassifier')

            self.run_hyperpipe(pipe, self.classification)

    def test_classification_12(self):
        X, y = load_iris(return_X_y=True)
        # multiclass classification
        for original_hyperpipe in self.hyperpipes:
            pipe = original_hyperpipe.copy_me()
            # Simple estimator Stack (train Random Forest on estimator stack proba outputs)
            # create estimator stack
            SVC1 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVC2 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestClassifier')
            # add to pipe
            pipe += Stack('estimator_stack', elements=[SVC1, SVC2, RF], use_probabilities=True)
            pipe += PipelineElement('RandomForestClassifier')

            pipe.optimization.metrics = ['accuracy']
            pipe.optimization.best_config_metric = 'accuracy'

            pipe.fit(X, y)
