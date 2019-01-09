import unittest

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.datasets import load_breast_cancer

from photonai.base.PhotonBase import *
from photonai.test.PipelineTests import DummyYAndCovariatesTransformer

import numpy as np


class HyperpipeTests(unittest.TestCase):

    def setUp(self):
        self.ss_pipe_element = PipelineElement('StandardScaler')
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.cv_object = KFold(n_splits=3)
        self.hyperpipe = Hyperpipe('god', inner_cv=self.cv_object, metrics=["accuracy"],
                                   best_config_metric="accuracy")
        self.hyperpipe += self.ss_pipe_element
        self.hyperpipe += self.pca_pipe_element
        self.hyperpipe.add(self.svc_pipe_element)

        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target

    def test_init(self):
        self.assertEqual(self.hyperpipe.name, 'god')
        # assure pipeline has two steps, first the pca and second the svc
        self.assertEqual(len(self.hyperpipe._pipe.steps), 3)
        self.assertIs(self.hyperpipe._pipe.steps[0][1], self.ss_pipe_element)
        self.assertIs(self.hyperpipe._pipe.steps[1][1], self.pca_pipe_element)
        self.assertIs(self.hyperpipe._pipe.steps[2][1], self.svc_pipe_element)

    def test_no_metrics(self):
        with self.assertRaises(ValueError):
            hyperpipe = Hyperpipe("hp_name", inner_cv=self.cv_object)

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
                            persist_options=PersistOptions(save_predictions='all'))

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

        my_pipe._generate_outer_cv_indices()
        tmp_counter = 0
        for train_idx_arr, test_idx_arr in my_pipe.data_test_cases:

            sk_results = {'accuracy': [], 'precision': [], 'f1_score': []}

            outer_train_X = self.__X[train_idx_arr]
            outer_train_y = self.__y[train_idx_arr]
            outer_test_X = self.__X[test_idx_arr]
            outer_test_y = self.__y[test_idx_arr]

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
            photon_pred = my_pipe.result_tree.outer_folds[tmp_counter].best_config.inner_folds[0].validation.y_pred
            self.assertTrue(np.array_equal(sk_prediction, photon_pred))

            # Check metrics
            for metric_name, metric_value in my_pipe.result_tree.outer_folds[tmp_counter].best_config.inner_folds[
                0].validation.metrics.items():
                self.assertEqual(sk_results[metric_name], metric_value)

            tmp_counter += 1


    def test_preprocessing(self):

        prepro_pipe = Preprocessing()
        prepro_pipe += PipelineElement.create("dummy", DummyYAndCovariatesTransformer(), {})

        self.hyperpipe += prepro_pipe
        self.hyperpipe.fit(self.__X, self.__y)

        self.assertTrue(np.array_equal(self.__y + 1, self.hyperpipe.y))


class PipelineElementTests(unittest.TestCase):

    def setUp(self):
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})

    def tearDown(self):
        pass

    def test_create_failure(self):
        with self.assertRaises(NameError):
            PipelineElement('NONSENSEName', {})

    def test_pipeline_element_create(self):
        # test name, set_disabled and base_element
        self.assertIsInstance(self.pca_pipe_element.base_element, PCA)

        # set_disabled is passed correctly
        self.assertTrue(self.pca_pipe_element.test_disabled)

        # correct name
        self.assertEqual(self.pca_pipe_element.name, 'PCA')

    def test_one_hyperparameter_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.pca_pipe_element.hyperparameters, {'PCA__n_components': [1, 2],
                                                                     'PCA__disabled': [False, True]})

        # config_grid is created as expected
        self.assertListEqual(self.pca_pipe_element.generate_config_grid(), [{'PCA__n_components': 1,
                                                                             'PCA__disabled': False},
                                                                            {'PCA__n_components': 2,
                                                                             'PCA__disabled': False},
                                                                            {'PCA__disabled': True}])

    def test_more_hyperparameters_setup(self):
        # sklearn attributes are generated
        self.assertDictEqual(self.svc_pipe_element.hyperparameters, {'SVC__C': [0.1, 1],
                                                                     'SVC__kernel': ['rbf', 'sigmoid']})

        # config_grid is created as expected
        self.assertListEqual(self.svc_pipe_element.generate_config_grid(), [{'SVC__C': 0.1, 'SVC__kernel': 'rbf'},
                                                                            {'SVC__C': 0.1, 'SVC__kernel': 'sigmoid'},
                                                                            {'SVC__C': 1, 'SVC__kernel': 'rbf'},
                                                                            {'SVC__C': 1, 'SVC__kernel': 'sigmoid'}])


    def test_set_params(self):
        config = {'n_components': 3, 'disabled': False}
        self.pca_pipe_element.set_params(**config)
        self.assertFalse(self.pca_pipe_element.disabled)
        self.assertEqual(self.pca_pipe_element.base_element.n_components, 3)
        with self.assertRaises(ValueError):
            self.pca_pipe_element.set_params(**{'any_weird_param': 1})

    def test_fit(self):
        pass


class PipelineSwitchTests(unittest.TestCase):

    def setUp(self):
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']})
        self.lr_pipe_element = PipelineElement('DecisionTreeClassifier', {'min_samples_split': [2, 3, 4]})
        self.pipe_switch = PipelineSwitch('switch', [self.svc_pipe_element, self.lr_pipe_element])

    def test_init(self):
        self.assertEqual(self.pipe_switch.name, 'switch')

    def test_hyperparams(self):
        # assert number of different configs to test
        # each config combi for each element: 4 for SVC and 3 for logistic regression = 7
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations), 2)
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations[0]), 4)
        self.assertEqual(len(self.pipe_switch.pipeline_element_configurations[1]), 3)

        # hyperparameters
        self.assertDictEqual(self.pipe_switch.hyperparameters, {'switch__current_element': [(0, 0), (0, 1),
                                                                                            (0, 2), (0, 3),
                                                                                            (1, 0), (1, 1),
                                                                                            (1, 2)]})

        # config grid
        self.assertListEqual(self.pipe_switch.generate_config_grid(), [{'switch__current_element': (0, 0)},
                                                                       {'switch__current_element': (0, 1)},
                                                                       {'switch__current_element': (0, 2)},
                                                                       {'switch__current_element': (0, 3)},
                                                                       {'switch__current_element': (1, 0)},
                                                                       {'switch__current_element': (1, 1)},
                                                                       {'switch__current_element': (1, 2)}])


    def test_set_params(self):
        false_config = {'current_element': 1}
        with self.assertRaises(ValueError):
            self.pipe_switch.set_params(**false_config)

        correct_config = {'current_element': (0, 1)}
        self.pipe_switch.set_params(**correct_config)
        self.assertEqual(self.pipe_switch.base_element.base_element.C, 0.1)
        self.assertEqual(self.pipe_switch.base_element.base_element.kernel, 'sigmoid')

    def test_base_element(self):
        self.pipe_switch.set_params(**{'switch__current_element': (1, 1)})
        self.assertIs(self.pipe_switch.base_element, self.lr_pipe_element)
        self.assertIs(self.pipe_switch.base_element.base_element, self.lr_pipe_element.base_element)


class PipelineBranchTests(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_breast_cancer(True)
        self.ss_pipe_element = PipelineElement("StandardScaler")
        self.pca_pipe_element = PipelineElement('PCA', {'n_components': [1, 2]}, test_disabled=True, random_state=3)
        self.svc_pipe_element = PipelineElement('SVC', {'C': [0.1, 1], 'kernel': ['rbf', 'sigmoid']}, random_state=3)

    def test_easy_use_case(self):
        sk_pipe = SKPipeline([("SS", StandardScaler()), ("PCA", PCA(random_state=3)), ("SVC", SVC(random_state=3))])
        sk_pipe.fit(self.X, self.y)
        sk_pred = sk_pipe.predict(self.X)

        branch = PipelineBranch("my_amazing_branch")
        branch += self.ss_pipe_element
        branch += self.pca_pipe_element
        branch += self.svc_pipe_element
        branch.fit(self.X, self.y)
        branch_pred = branch.predict(self.X)

        self.assertTrue(np.array_equal(sk_pred, branch_pred))

    def test_no_y_transformers(self):
        branch = PipelineBranch("forbidden_branch")
        stacking_element = PipelineStacking("forbidden_stack")
        my_dummy = PipelineElement.create("dummy", DummyYAndCovariatesTransformer(), {})
        with self.assertRaises(ValueError):
            branch += my_dummy

        with self.assertRaises(ValueError):
            stacking_element += my_dummy

    def test_stacking_of_branches(self):
        branch1 = PipelineBranch("B1")
        branch1.add(PipelineElement("StandardScaler"))

        branch2 = PipelineBranch("B2")
        branch2.add(PipelineElement("PCA", random_state=3))

        stacking_element = PipelineStacking("Stack")
        stacking_element += branch1
        stacking_element += branch2

        stacking_element.fit(self.X, self.y)
        trans = stacking_element.transform(self.X)
        pred = stacking_element.predict(self.X)

        self.assertTrue(np.array_equal(trans, pred))

        ss = StandardScaler()
        pca = PCA(random_state=3)
        ss.fit(self.X, self.y)
        X_ss = ss.transform(self.X)
        pca.fit(self.X, self.y)
        X_pca = pca.transform(self.X)

        matrix = np.concatenate((X_ss, X_pca), axis=1)

        self.assertTrue(np.array_equal(trans, matrix))


    def test_voting(self):

        svc1 = PipelineElement("SVC", random_state=1)
        svc2 = PipelineElement("SVC", random_state=1)
        stack_obj = PipelineStacking("StackItem", voting=True)
        stack_obj += svc1
        stack_obj += svc2

        sk_svc1 = SVC()
        sk_svc2 = SVC()
        pass



if __name__ == '__main__':
    unittest.main()


