import random
import unittest

from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score

from photonai.base.PhotonBase import PipelineElement, Hyperpipe


class CVTestsCaseA(unittest.TestCase):
    __X = None
    __y = None

    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target
        random.seed(42)

    def testCaseA(self):
        pca_n_components = 10
        svc_c = 1
        svc_kernel = "rbf"
        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                            metrics=['accuracy', 'precision', 'f1_score'],
                            inner_cv=KFold(n_splits=3),
                            outer_cv=KFold(n_splits=3),
                            eval_final_performance=True)

        my_pipe += PipelineElement('StandardScaler')
        my_pipe += PipelineElement('PCA', {'n_components': [pca_n_components]}, random_state=3)
        my_pipe += PipelineElement('SVC', {'C': [svc_c], 'kernel': [svc_kernel]}, random_state=3)

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)

        # print(my_pipe._test_performances)
        # from Framework import LogExtractor
        # log_ex = LogExtractor.LogExtractor(my_pipe.result_tree)
        # log_ex.extract_csv("test_case_A.csv")

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

            for metric in my_pipe.result_tree.outer_folds[tmp_counter].tested_config_list[0].metrics_test:
                if metric.operation == "FoldOperations.MEAN":
                    self.assertEqual(sk_results[metric.metric_name], metric.value)

            tmp_counter += 1



if __name__ == '__main__':
    unittest.main()
