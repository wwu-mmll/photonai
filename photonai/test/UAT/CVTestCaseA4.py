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
        print(self.__X.shape)
        random.seed(42)

    def testCaseA(self):
        pca_n_components = [2, 5]
        svc_c = [.1, 1, 5]
        #svc_kernel = ['rbf']
        svc_kernel = ['rbf','linear']

        # SET UP HYPERPIPE
        my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                            optimizer_params={},
                            metrics=['accuracy', 'precision', 'f1_score'],
                            inner_cv=KFold(
                                n_splits=2, random_state=3), eval_final_performance=False)

        my_pipe += PipelineElement.create('standard_scaler')
        my_pipe += PipelineElement.create('pca', {'n_components': pca_n_components})
        my_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})

        # START HYPERPARAMETER SEARCH
        my_pipe.fit(self.__X, self.__y)
        print(my_pipe._test_performances)
        pipe_results = {'train': [], 'test': []}
        for i in range(len(my_pipe._performance_history_list)):
            pipe_results['train'].extend(
                my_pipe._performance_history_list[i]['accuracy_folds']['train'])
            pipe_results['test'].extend(
                my_pipe._performance_history_list[i]['accuracy_folds']['test'])

        print('\n\n')
        print('Running sklearn version...')
        #cv_outer = KFold(n_splits=2, random_state=3)
        cv_inner_1 = KFold(n_splits=2, random_state=3)


        sk_results = {'train': [], 'test': []}

        for n_comp in pca_n_components:
            for c in svc_c:
                for current_kernel in svc_kernel:
                    tr_acc = []
                    val_acc = []
                    for train_2, val_1 in cv_inner_1.split(self.__X):

                        data_train_2 = self.__X[train_2]
                        print(data_train_2.shape)
                        data_val_1 = self.__X[val_1]
                        y_train_2 = self.__y[train_2]
                        y_val_1 = self.__y[val_1]

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
        print('SkL  test: ', sk_results['test'])
        print('Pipe test: ', pipe_results['test'])

        self.assertEqual(sk_results['test'], pipe_results['test'])
        self.assertEqual(sk_results['train'], pipe_results['train'])


if __name__ == '__main__':
    unittest.main()
