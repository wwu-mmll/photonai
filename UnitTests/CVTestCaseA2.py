import unittest
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from HPOFramework.HPOBaseClasses import PipelineElement, Hyperpipe
from sklearn.model_selection._validation import _fit_and_score
import random
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class CVTestsLocalSearchTrue(unittest.TestCase):
    __X = None
    __y = None

    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target
        random.seed(42)

    def testCaseA(self):
        pca_n_components = [2, 3, 10]
        svc_c = [.1, 1, 10]
        svc_kernel = ['rbf']
        #svc_kernel = ['rbf','linear']

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search', optimizer_params={},
                            metrics=['accuracy', 'precision', 'f1_score'],
                            hyperparameter_specific_config_cv_object=KFold(n_splits=3, random_state=3),
                            hyperparameter_search_cv_object=KFold(n_splits=3, random_state=3))

        my_pipe += PipelineElement.create('standard_scaler')
        #my_pipe += PipelineElement.create('pca', {'n_components': pca_n_components})
        my_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)
        print(my_pipe.test_performances)



        # SKLearn Implementation using Loops
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import f1_score, accuracy_score, precision_score

        print('\n\n\n')
        print('Running sklearn version...')
        cv_outer = KFold(n_splits=3, random_state=3)
        cv_inner_1 = KFold(n_splits=3, random_state=3)

        for n_comp in pca_n_components:
            for current_kernel in svc_kernel:
                for c in svc_c:
                    tr_acc = []
                    val_acc = []

                    for train_1, test in cv_outer.split(self.__X):
                        data_train_1 = self.__X[train_1]
                        data_test = self.__X[test]
                        y_train_1 = self.__y[train_1]
                        y_test = self.__y[test]

                        for train_2, val_1 in cv_inner_1.split(data_train_1):
                            data_train_2 = data_train_1[train_2]
                            data_val_1 = data_train_1[val_1]
                            y_train_2 = y_train_1[train_2]
                            y_val_1 = y_train_1[val_1]


                            #my_pca = PCA(n_components=n_comp)
                            #my_pca.fit(data_train_2)
                            #data_tr_2_pca = my_pca.transform(data_train_2)
                            #data_val_1_pca = my_pca.transform(data_val_1)

                            data_tr_2_pca = data_train_2
                            data_val_1_pca = data_val_1

                            my_svc = SVC(kernel=current_kernel, C=c)
                            my_svc.fit(data_tr_2_pca,y_train_2)

                            tr_acc.append(my_svc.score(data_tr_2_pca,y_train_2))
                            val_acc.append(my_svc.score(data_val_1_pca,y_val_1))
                            print('n_components: ', n_comp)
                            print('kernel: ', current_kernel)
                            print('c: ', c)
                            print('Training 2 Accuracy: ', tr_acc[-1])
                            print('Validation 1 Accuracy: ', val_acc[-1])

                        print('Done with current HP set...')
                        #print('Training: ', tr_acc)
                        #print('Validation: ', val_acc)

        # self.assertEqual(sk_results['accuracy'], my_pipe.test_performances['accuracy'][tmp_counter])
        # self.assertEqual(sk_results['precision'], my_pipe.test_performances['precision'][tmp_counter])
        # self.assertEqual(sk_results['f1_score'], my_pipe.test_performances['f1_score'][tmp_counter])
        #
        # tmp_counter += 1


    # def score(self, estimator, X, y_true):
    #     default_score = estimator.score(X, y_true)
    #     return default_score


if __name__ == '__main__':
    unittest.main()
