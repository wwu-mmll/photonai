import unittest
import numpy as np

from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

from photonai.base import PipelineElement, Hyperpipe, OutputSettings, Preprocessing
from photonai.test.base.photon_pipeline_tests import DummyYAndCovariatesTransformer


class HyperpipeTests(unittest.TestCase):

    def setUp(self):
        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.cv_object = KFold(n_splits=3)
        self.hyperpipe = Hyperpipe('god', inner_cv=self.cv_object, metrics=["accuracy"],
                                   best_config_metric="accuracy",
                                   output_settings=OutputSettings(project_folder='./tmp'))
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
                            output_settings=OutputSettings(save_predictions='all', project_folder='./tmp'))

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
