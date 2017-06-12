# Case B: Nested Hyperparameter Optimization with Local Optimization
# First, find "best" config for PCA (HyperPipe2) and second, find best
# config for SVM

import unittest
from sklearn.model_selection import KFold, ShuffleSplit
from HPOFramework.HPOBaseClasses import PipelineElement, Hyperpipe, PipelineFusion
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
np.random.seed(3)
from sklearn.metrics import mean_absolute_error as mae

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
        #svc_kernel = ['rbf']
        svc_kernel = ['rbf','linear']

        # SET UP HYPERPIPE
        # outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
        #                     metrics=['accuracy'],
        #                     hyperparameter_specific_config_cv_object=KFold(n_splits=2, random_state=3),
        #                     hyperparameter_search_cv_object=KFold(n_splits=3, random_state=3),
        #                     eval_final_performance=True)
        outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
                               metrics=['accuracy'],
                               hyperparameter_specific_config_cv_object=KFold(
                                   n_splits=2, random_state=3),
                               hyperparameter_search_cv_object=ShuffleSplit(n_splits=1,test_size=0.2),
                               eval_final_performance=True)
        inner_pipe = Hyperpipe('pca_pipe', optimizer='grid_search',
                               hyperparameter_specific_config_cv_object=KFold(n_splits=2, random_state=3),
                               eval_final_performance=False)

        inner_pipe.add(PipelineElement.create('standard_scaler'))
        inner_pipe.add(PipelineElement.create('ae_pca', {'n_components': pca_n_components}))

        pipeline_fusion = PipelineFusion('fusion_element', [inner_pipe])

        outer_pipe.add(pipeline_fusion)
        outer_pipe.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))

        # START HYPERPARAMETER SEARCH
        outer_pipe.fit(self.__X, self.__y)
        print(outer_pipe.test_performances)
        pipe_results = {'train': [], 'test': []}
        for i in range(len(outer_pipe.performance_history_list)):
            pipe_results['train'].extend(
                outer_pipe.performance_history_list[i]['accuracy_folds']['train'])
            pipe_results['test'].extend(
                outer_pipe.performance_history_list[i]['accuracy_folds']['test'])

        print(outer_pipe.test_performances['accuracy'])

        print('\n\n')
        print('Running sklearn version...\n')
        #cv_outer = KFold(n_splits=3, random_state=3)
        cv_outer = ShuffleSplit(n_splits=1,test_size=0.2)
        cv_inner_1 = KFold(n_splits=2, random_state=3)
        cv_inner_2 = KFold(n_splits=2, random_state=3)

        opt_tr_acc = []
        opt_test_acc = []
        fold_cnt = 1
        for train_1, test in cv_outer.split(self.__X):
            data_train_1 = self.__X[train_1]
            data_test = self.__X[test]
            y_train_1 = self.__y[train_1]
            y_test = self.__y[test]
            sk_results = {'train_1': [], 'test': [], 'train_1_mean': [],
                          'test_mean': []}
            config_inner_1 = {'C':[], 'kernel':[]}
            sk_results_inner1 = {'train_2': [],
                                 'val_1': [],
                                 'train_2_mean': [],
                                 'val_1_mean': []}
            print('\n\nSklearn Outer Pipe Fold', fold_cnt,'\n')
            for c in svc_c:
                for current_kernel in svc_kernel:
                    config_inner_1['C'].extend([c])
                    config_inner_1['kernel'].extend([current_kernel])
                    svc_score_tr = []
                    svc_score_te = []

                    print('C:', c, 'Kernel:', current_kernel)
                    for train_2, val_1 in cv_inner_1.split(
                            data_train_1):
                        data_train_2 = data_train_1[train_2]
                        data_val_1 = data_train_1[val_1]
                        y_train_2 = y_train_1[train_2]
                        y_val_1 = y_train_1[val_1]

                        config_inner_2 = {'n_comp': []}

                        for n_comp in pca_n_components:
                            config_inner_2['n_comp'].extend([n_comp])

                            tr_acc = []
                            val_acc = []
                            sk_results_inner2 = {'train_3': [], 'val_2': [],
                                          'train_3_mean': [],
                                          'val_2_mean': []}
                            # print('Some training data:',
                            #       data_train_2[0:2, 0:2])
                            for train_3, val_2 in cv_inner_2.split(
                                    data_train_2):

                                data_train_3 = data_train_2[train_3]

                                data_val_2 = data_train_2[val_2]
                                y_train_3 = y_train_2[train_3]
                                y_val_2 = y_train_2[val_2]

                                my_scaler = StandardScaler()
                                my_scaler.fit(data_train_3)
                                data_train_3 = my_scaler.transform(data_train_3)
                                data_val_2 = my_scaler.transform(data_val_2)

                                # Run PCA
                                my_pca = PCA(n_components=n_comp)
                                my_pca.fit(data_train_3)
                                data_tr_3_pca = my_pca.transform(data_train_3)
                                data_val_2_pca = my_pca.transform(data_val_2)

                                data_tr_3_pca_inv = my_pca.inverse_transform(data_tr_3_pca)
                                data_val_2_pca_inv = my_pca.inverse_transform(data_val_2_pca)

                                mae_tr = mae(data_train_3,data_tr_3_pca_inv)
                                mae_te = mae(data_val_2,data_val_2_pca_inv)

                                tr_acc.append(mae_tr)
                                val_acc.append(mae_te)

                            sk_results_inner2['train_3'].extend(tr_acc)
                            sk_results_inner2['val_2'].extend(val_acc)
                            sk_results_inner2['train_3_mean'].extend([np.mean(tr_acc)])
                            sk_results_inner2['val_2_mean'].extend([np.mean(val_acc)])
                            print('Sklearn PCA Pipe')
                            print('n_comp:', n_comp)
                            print('Training 3:', [np.mean(tr_acc)],
                                  'Validation 2:', [np.mean(val_acc)])
                        # find best config for val 2
                        best_config_id = np.argmax(sk_results_inner2['val_2_mean'])
                        print('Best config:', config_inner_2['n_comp'][best_config_id], '\n')
                        # fit optimum pipe
                        my_scaler = StandardScaler()
                        my_scaler.fit(data_train_2)
                        data_train_2 = my_scaler.transform(data_train_2)
                        data_val_1 = my_scaler.transform(data_val_1)

                        # Run PCA
                        my_pca = PCA(n_components=config_inner_2['n_comp'][best_config_id])
                        my_pca.fit(data_train_2)
                        data_tr_2_pca = my_pca.transform(data_train_2)
                        data_val_1_pca = my_pca.transform(data_val_1)

                        # Run SVC
                        my_svc = SVC(kernel=current_kernel, C=c)
                        my_svc.fit(data_tr_2_pca, y_train_2)
                        svc_score_tr.append(my_svc.score(data_tr_2_pca, y_train_2))
                        svc_score_te.append(my_svc.score(data_val_1_pca, y_val_1))

                    sk_results_inner1['train_2'].extend(svc_score_tr)
                    sk_results_inner1['val_1'].extend(svc_score_te)
                    sk_results_inner1['train_2_mean'].extend([np.mean(svc_score_tr)])
                    sk_results_inner1['val_1_mean'].extend([np.mean(svc_score_te)])

                fold_cnt += 1
            best_config_id_inner_1 = np.argmax(sk_results_inner1['val_1_mean'])

            # fit optimum pipe
            my_scaler = StandardScaler()
            my_scaler.fit(data_train_1)
            data_train_1 = my_scaler.transform(data_train_1)
            data_test = my_scaler.transform(data_test)

            # Run PCA
            my_pca = PCA(n_components=config_inner_2['n_comp'][best_config_id])
            my_pca.fit(data_train_1)
            data_tr_1_pca = my_pca.transform(data_train_1)
            data_test_pca = my_pca.transform(data_test)

            # Run SVC
            my_svc = SVC(kernel=config_inner_1['kernel'][best_config_id_inner_1],
                         C=config_inner_1['C'][best_config_id_inner_1])
            print('Best Config:...')
            print('C = ',config_inner_1['C'][best_config_id_inner_1])
            print('kernel=', config_inner_1['kernel'][best_config_id_inner_1])

            my_svc.fit(data_tr_1_pca, y_train_1)

            opt_tr_acc.append(my_svc.score(data_tr_1_pca, y_train_1))
            opt_test_acc.append(my_svc.score(data_test_pca, y_test))


        print('\nCompare results of last iteration (outer cv)...')
        print('SkL  Train:', sk_results_inner1['train_2'])
        print('Pipe Train:', pipe_results['train'])
        print('SkL  Test: ', sk_results_inner1['val_1'])
        print('Pipe Test: ', pipe_results['test'])
        print('\nEval final performance:')
        print('Pipe final perf:', outer_pipe.test_performances['accuracy'])
        print('Sklearn final perf:', opt_test_acc)
        # self.assertEqual(sk_results_inner1['train_2'], pipe_results['train'])
        # self.assertEqual(sk_results_inner1['val_1'], pipe_results['test'])
        self.assertEqual(opt_test_acc, outer_pipe.test_performances['accuracy'])


if __name__ == '__main__':
    unittest.main()