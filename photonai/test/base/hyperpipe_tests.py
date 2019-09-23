import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import PipelineElement, Hyperpipe, OutputSettings, Preprocessing, CallbackElement, Branch, Stack
from photonai.optimization import IntegerRange, Categorical
from photonai.test.PhotonBaseTest import PhotonBaseTest
from photonai.test.base.dummy_elements import DummyTransformer
from photonai.test.base.photon_elements_tests import elements_to_dict
from photonai.test.base.photon_pipeline_tests import DummyYAndCovariatesTransformer


class HyperpipeTests(PhotonBaseTest):

    def setUp(self):

        super(HyperpipeTests, self).setUp()
        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.cv_object = KFold(n_splits=3)
        self.hyperpipe = Hyperpipe('god', inner_cv=self.cv_object, metrics=["accuracy"],
                                   best_config_metric="accuracy",
                                   output_settings=OutputSettings(project_folder='./tmp', overwrite_results=True))
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_init(self):
        self.assertEqual(self.hyperpipe.name, 'god')
        # assure pipeline has two elements, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe._pipe.elements), 3)
        self.assertIs(self.hyperpipe._pipe.elements[0][1], self.ss_pipe_element)
        self.assertIs(self.hyperpipe._pipe.elements[1][1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe._pipe.elements[2][1], self.svc_pipe_element)

    def test_no_metrics(self):
        # make sure that no metrics means raising an error
        with self.assertRaises(ValueError):
            hyperpipe = Hyperpipe("hp_name", inner_cv=self.cv_object)

        # make sure that if no best config metric is given, PHOTON raises a warning
        with self.assertRaises(Warning):
            hyperpipe = Hyperpipe("hp_name", inner_cv=self.cv_object, metrics=["accuracy", "f1_score"])

    def test_easy_use_case(self):

        pca_n_components = 10
        svc_c = 1
        svc_kernel = "rbf"

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                            metrics=['accuracy', 'precision', 'f1_score'],
                            best_config_metric='accuracy',
                            inner_cv=KFold(n_splits=3),
                            outer_cv=KFold(n_splits=3),
                            eval_final_performance=True,
                            output_settings=OutputSettings(save_predictions='all', project_folder='./tmp'))

        my_pipe += PipelineElement('StandardScaler')
        my_pipe += PipelineElement('PCA', {'n_components': [pca_n_components]}, random_state=3)
        my_pipe += PipelineElement('SVC', {'C': [svc_c], 'kernel': [svc_kernel]}, random_state=3)

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)

        # Das muss noch weg! ToDo
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import f1_score, accuracy_score, precision_score

        # Now we are using the native Scikit-learn methods
        sk_pipeline = Pipeline([("standard_scaler", StandardScaler()),
                                ("pca", PCA(n_components=pca_n_components, random_state=3)),
                                ("svc", SVC(C=svc_c, kernel=svc_kernel, random_state=3))])

        tmp_counter = 0
        for outer_fold in list(my_pipe.cross_validation.outer_folds.values()):

            sk_results = {'accuracy': [], 'precision': [], 'f1_score': []}

            outer_train_X = self.__X[outer_fold.train_indices]
            outer_train_y = self.__y[outer_fold.train_indices]
            outer_test_X = self.__X[outer_fold.test_indices]
            outer_test_y = self.__y[outer_fold.test_indices]

            sk_config_cv = KFold(n_splits=3)
            # # Todo: test other configs and select best!
            # for sub_train_idx, sub_test_idx in sk_config_cv.split(outer_train_X, outer_train_y):
            #     inner_train_X = self.__X[sub_train_idx]
            #     inner_train_y = self.__y[sub_train_idx]
            #     #test_X = self.__X[sub_test_idx]
            #     #test_y = self.__y[sub_test_idx]
            #
            #     # sk_pipeline.fit(inner_train_X, inner_train_y)
            #
            #     fit_and_predict_score = _fit_and_score(sk_pipeline, outer_train_X, outer_train_y, self.score,
            #                                            sub_train_idx, sub_test_idx, verbose=0, parameters={},
            #                                            fit_params={},
            #                                            return_train_score=True,
            #                                            return_n_test_samples=True,
            #                                            return_times=True, return_parameters=True,
            #                                            error_score='raise')

            sk_pipeline.fit(outer_train_X, outer_train_y)
            sk_prediction = sk_pipeline.predict(outer_test_X)

            sk_results['accuracy'].append(accuracy_score(outer_test_y, sk_prediction))
            sk_results['precision'].append(precision_score(outer_test_y, sk_prediction))
            sk_results['f1_score'].append(f1_score(outer_test_y, sk_prediction))

            # bestItem = np.argmax(sk_results['default'])
            # print([str(k)+':'+str(i[bestItem]) for k, i in sk_results.items()])

            # Check prediction arrays
            photon_pred = my_pipe.results.outer_folds[tmp_counter].best_config.best_config_score.validation.y_pred
            self.assertTrue(np.array_equal(sk_prediction, photon_pred))

            # Check metrics
            for metric_name, metric_value in my_pipe.results.outer_folds[tmp_counter].best_config.best_config_score.validation.metrics.items():
                self.assertEqual(sk_results[metric_name], metric_value)

            tmp_counter += 1

    def test_eval_final_performance(self):

        pca_n_components = 10
        svc_c = 1
        svc_kernel = "rbf"

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                            metrics=['accuracy', 'precision', 'f1_score'],
                            best_config_metric='accuracy',
                            inner_cv=KFold(n_splits=3),
                            # outer_cv=KFold(n_splits=3),
                            eval_final_performance=True,
                            output_settings=OutputSettings(save_predictions='all', project_folder='./tmp',
                                                           overwrite_results=True))

        my_pipe += PipelineElement('StandardScaler')
        my_pipe += PipelineElement('PCA', {'n_components': [pca_n_components]}, random_state=3)
        my_pipe += PipelineElement('SVC', {'C': [svc_c], 'kernel': [svc_kernel]}, random_state=3)

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)

        # Todo: check that no outer fold metrics have been created

    def test_preprocessing(self):

        prepro_pipe = Preprocessing()
        prepro_pipe += PipelineElement.create("dummy", DummyYAndCovariatesTransformer(), {})

        self.hyperpipe += prepro_pipe
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(np.array_equal(self.__y + 1, self.hyperpipe.data.y))

    def test_estimation_type(self):
        def callback(X, y=None, **kwargs):
            pass

        pipe = Hyperpipe('name', inner_cv=KFold(n_splits=2), best_config_metric='mean_squared_error')

        with self.assertRaises(NotImplementedError):
            pipe += PipelineElement('PCA')
            est_type = pipe.estimation_type

        pipe += PipelineElement('SVC')
        self.assertEqual(pipe.estimation_type, 'classifier')

        pipe.elements[-1] = PipelineElement('SVR')
        self.assertEqual(pipe.estimation_type, 'regressor')

        with self.assertRaises(NotImplementedError):
            pipe.elements[-1] = CallbackElement('MyCallback', callback)
            est_type = pipe.estimation_type

    def test_copy_me(self):
        self.maxDiff = None
        copy = self.hyperpipe.copy_me()
        copy2 = self.hyperpipe.copy_me()
        self.assertDictEqual(elements_to_dict(copy), elements_to_dict(self.hyperpipe))

        copy_after_fit = self.hyperpipe.fit(self.__X, self.__y).copy_me()

        copy_after_fit = elements_to_dict(copy_after_fit)
        # the current_configs of the elements are not None after calling fit() on a hyperpipe
        # when copying the respective PipelineElement, these current_configs are copied, too
        # this is why we need to delete _pipe and elements before asserting for equality
        copy_after_fit['_pipe'] = None
        copy_after_fit['elements'] = None
        copy = elements_to_dict(copy)
        copy['_pipe'] = None
        copy['elements'] = None
        self.assertDictEqual(copy, copy_after_fit)

        # check if deepcopy worked
        copy2.cross_validation.inner_cv.n_splits = 10
        self.assertEqual(copy2.cross_validation.inner_cv.n_splits, 10)
        self.assertEqual(self.hyperpipe.cross_validation.inner_cv.n_splits, 3)

    def recursive_assertion(self, element_a, element_b):
        if isinstance(element_a, dict):
            for key in element_a.keys():
                self.recursive_assertion(element_a[key], element_b[key])
        elif isinstance(element_a, np.ndarray):
            np.testing.assert_array_equal(element_a, element_b)
        elif isinstance(element_a, list):
            for i, _ in enumerate(element_a):
                self.recursive_assertion(element_a[i], element_b[i])
        elif isinstance(element_a, tuple):
            for i in range(len(element_a)):
                self.recursive_assertion(element_a[i], element_b[i])
        else:
            self.assertEqual(element_a, element_b)

    def test_save_optimum_pipe(self):
        # todo: test .save() of custom model
        settings = OutputSettings(project_folder='./tmp/optimum_pipypipe/', overwrite_results=True)

        my_pipe = Hyperpipe('hyperpipe',
                            optimizer='random_grid_search',
                            optimizer_params={'k': 3},
                            metrics=['accuracy', 'precision', 'recall'],
                            best_config_metric='f1_score',
                            outer_cv=KFold(n_splits=2),
                            inner_cv=KFold(n_splits=2),
                            verbosity=1,
                            output_settings=settings)

        preproc = Preprocessing()
        preproc += PipelineElement('LabelEncoder')

        # BRANCH WITH QUANTILTRANSFORMER AND DECISIONTREECLASSIFIER
        tree_qua_branch = Branch('tree_branch')
        tree_qua_branch += PipelineElement('QuantileTransformer')
        tree_qua_branch += PipelineElement('DecisionTreeClassifier', {'min_samples_split': IntegerRange(2, 4)},
                                           criterion='gini')

        # BRANCH WITH MinMaxScaler AND DecisionTreeClassifier
        svm_mima_branch = Branch('svm_branch')
        svm_mima_branch += PipelineElement('MinMaxScaler')
        svm_mima_branch += PipelineElement('SVC', {'kernel': Categorical(['rbf', 'linear']), 'C': 2.0}, gamma='auto')

        # BRANCH WITH StandardScaler AND KNeighborsClassifier
        knn_sta_branch = Branch('neighbour_branch')
        knn_sta_branch += PipelineElement.create("dummy", DummyTransformer(), {})
        knn_sta_branch += PipelineElement('KNeighborsClassifier')

        # voting = True to mean the result of every branch
        my_pipe += Stack('final_stack', [tree_qua_branch, svm_mima_branch, knn_sta_branch])

        my_pipe += PipelineElement('LogisticRegression', solver='lbfgs')

        pipe_copy = my_pipe.copy_me()
        pipe_copy.output_settings.save_output = False
        pipe_copy.fit(self.__X, self.__y)
        self.assertFalse(os.path.exists("./tmp/optimum_pipypipe/hyperpipe_results/"))

        my_pipe.fit(self.__X, self.__y)
        self.assertTrue(os.path.exists("./tmp/optimum_pipypipe/hyperpipe_results/photon_best_model.photon"))

        # check if load_optimum_pipe also works
        loaded_optimum_pipe = Hyperpipe.load_optimum_pipe(
            "./tmp/optimum_pipypipe/hyperpipe_results/photon_best_model.photon")
        y_pred_loaded = loaded_optimum_pipe.predict(self.__X)
        y_pred = my_pipe.optimum_pipe.predict(self.__X)
        np.testing.assert_array_equal(y_pred_loaded, y_pred)
