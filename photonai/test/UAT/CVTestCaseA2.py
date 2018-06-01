import random
import unittest

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Framework.PhotonBase import PipelineElement, Hyperpipe


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
        pca_n_components = [2, 5]
        svc_c = [.1, 1]
        svc_kernel = ['rbf']
        # svc_kernel = ['rbf','linear']

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                            optimizer_params={},
                            inner_cv=KFold(
                                n_splits=2, random_state=3),
                            outer_cv=KFold(
                                n_splits=2, random_state=3), verbose=2, eval_final_performance=True)

        my_pipe += PipelineElement.create('standard_scaler')
        my_pipe += PipelineElement.create('pca', {'n_components': pca_n_components})
        my_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)
        from Framework import LogExtractor
        log_ex = LogExtractor.LogExtractor(my_pipe.result_tree)
        log_ex.extract_csv("test_case_A2.csv")

        # print(my_pipe.test_performances)
        # pipe_results = {'train': [], 'test': []}
        # for i in range(len(my_pipe.performance_history_list)):
        #     pipe_results['train'].extend(
        #         my_pipe.performance_history_list[i]['accuracy_folds']['train'])
        #     pipe_results['test'].extend(
        #         my_pipe.performance_history_list[i]['accuracy_folds']['test'])

        print('\n\n')
        print('Running sklearn version...')
        cv_outer = KFold(n_splits=2, random_state=3)
        cv_inner_1 = KFold(n_splits=2, random_state=3)

        for train_1, test in cv_outer.split(self.__X):
            data_train_1 = self.__X[train_1]
            data_test = self.__X[test]
            y_train_1 = self.__y[train_1]
            y_test = self.__y[test]
            sk_results = {'train': [], 'test': []}

            for n_comp in pca_n_components:
                for current_kernel in svc_kernel:
                    for c in svc_c:
                        tr_acc = []
                        val_acc = []

                        for train_2, val_1 in cv_inner_1.split(
                                data_train_1):
                            data_train_2 = data_train_1[train_2]
                            data_val_1 = data_train_1[val_1]
                            y_train_2 = y_train_1[train_2]
                            y_val_1 = y_train_1[val_1]

                            my_scaler = StandardScaler()
                            my_scaler.fit(data_train_2)
                            data_train_2 = my_scaler.transform(data_train_2)
                            data_val_1 = my_scaler.transform(data_val_1)

                            # Run PCA
                            my_pca = PCA(n_components=n_comp)
                            my_pca.fit(data_train_2)
                            data_tr_2_pca = my_pca.transform(data_train_2)
                            data_val_1_pca = my_pca.transform(data_val_1)

                            # Run SVC
                            my_svc = SVC(kernel=current_kernel, C=c)
                            my_svc.fit(data_tr_2_pca, y_train_2)

                            tr_acc.append(my_svc.score(data_tr_2_pca, y_train_2))
                            val_acc.append(my_svc.score(data_val_1_pca, y_val_1))
                            print('n_components: ', n_comp, 'kernel:',
                                  current_kernel, 'c:', c)
                            print('Training 2:', tr_acc[-1],
                                  'validation 1:', val_acc[-1])

                        sk_results['train'].extend(tr_acc)
                        sk_results['test'].extend(val_acc)

        print('\nCompare results of last iteration (outer cv)...')
        print('SkL  Train:', sk_results['train'])
        print('Pipe Train:', pipe_results['train'])
        print('SkL  Test: ', sk_results['test'])
        print('Pipe Test: ', pipe_results['test'])

        self.assertEqual(sk_results['test'], pipe_results['test'])
        self.assertEqual(sk_results['train'], pipe_results['train'])


if __name__ == '__main__':
    unittest.main()
