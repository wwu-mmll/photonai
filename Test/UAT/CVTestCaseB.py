# Case B: Nested Hyperparameter Optimization with Local Optimization
# First, find "best" config for PCA (HyperPipe2) and second, find best
# config for SVM

import random
import unittest

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineStacking
from PipelineWrapper.PCA_AE_Wrapper import PCA_AE_Wrapper

np.random.seed(3)


class CVTestCaseB(unittest.TestCase):
    __X = None
    __y = None

    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target
        random.seed(42)

    def testCaseB(self):
        pca_n_components = [7, 15, 10]
        svc_c = [.1, 1]
        #svc_kernel = ['rbf']
        svc_kernel = ['rbf', 'linear']
        cv_outer = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
        cv_inner_1 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
        cv_inner_2 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)

        # SET UP HYPERPIPE
        outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
                               metrics=['accuracy'],
                               inner_cv=cv_inner_1,
                               outer_cv=cv_outer,
                               eval_final_performance=True)
        inner_pipe = Hyperpipe('pca_pipe', optimizer='grid_search',
                               inner_cv=cv_inner_2,
                               eval_final_performance=False)

        inner_pipe.add(PipelineElement.create('standard_scaler'))
        inner_pipe.add(PipelineElement.create('ae_pca', {'n_components': pca_n_components}))

        pipeline_fusion = PipelineStacking('fusion_element', [inner_pipe])

        outer_pipe.add(pipeline_fusion)
        outer_pipe.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))

        # START HYPERPARAMETER SEARCH
        outer_pipe.fit(self.__X, self.__y)
        print(outer_pipe._test_performances)
        pipe_results = {'train': [], 'test': []}
        for i in range(len(outer_pipe._performance_history_list)):
            pipe_results['train'].extend(
                outer_pipe._performance_history_list[i]['accuracy_folds']['train'])
            pipe_results['test'].extend(
                outer_pipe._performance_history_list[i]['accuracy_folds']['test'])

        print(outer_pipe._test_performances['accuracy'])

        print('\n\n')
        print('Running sklearn version...\n')
        opt_tr_acc = []
        opt_test_acc = []

        for train_1, test in cv_outer.split(self.__X):
            data_train_1 = self.__X[train_1]
            data_test = self.__X[test]
            y_train_1 = self.__y[train_1]
            y_test = self.__y[test]
            config_inner_1 = {'C':[], 'kernel':[]}
            sk_results_inner1 = {'train_2': [],
                                 'val_1': [],
                                 'train_2_mean': [],
                                 'val_1_mean': []}
            print('Outer Split')
            print('n train_1:', data_train_1.shape[0], '\n')


            for c in svc_c:
                for current_kernel in svc_kernel:
                    config_inner_1['C'].extend([c])
                    config_inner_1['kernel'].extend([current_kernel])

                    print('C:', c, 'Kernel:', current_kernel, '\n')
                    svc_score_tr = []
                    svc_score_te = []
                    fold_cnt = 1
                    for train_2, val_1 in cv_inner_1.split(
                            data_train_1):
                        print('\n\nSklearn Outer Pipe FoldMetrics', fold_cnt)

                        data_train_2 = data_train_1[train_2]
                        data_val_1 = data_train_1[val_1]
                        y_train_2 = y_train_1[train_2]
                        y_val_1 = y_train_1[val_1]
                        print('n train_2:', data_train_2.shape[0], '\n')


                        config_inner_2 = {'n_comp': []}
                        print('Sklearn PCA Pipe')
                        sk_results_inner2 = {'train_3': [], 'val_2': [],
                                             'train_3_mean': [],
                                             'val_2_mean': []}
                        for n_comp in pca_n_components:
                            config_inner_2['n_comp'].extend([n_comp])

                            tr_acc = []
                            val_acc = []

                            # print('Some training data:',
                            #       data_train_2[0:2, 0:2])
                            for train_3, val_2 in cv_inner_2.split(
                                    data_train_2):

                                data_train_3 = data_train_2[train_3]
                                data_val_2 = data_train_2[val_2]

                                my_scaler = StandardScaler()
                                my_scaler.fit(data_train_3)
                                data_train_3 = my_scaler.transform(data_train_3)
                                data_val_2 = my_scaler.transform(data_val_2)

                                # Run PCA
                                my_pca = PCA_AE_Wrapper(n_components=n_comp)
                                my_pca.fit(data_train_3)

                                mae_tr = my_pca.score(data_train_3)
                                mae_te = my_pca.score(data_val_2)

                                tr_acc.append(mae_tr)
                                val_acc.append(mae_te)

                            sk_results_inner2['train_3'].extend(tr_acc)
                            sk_results_inner2['val_2'].extend(val_acc)
                            sk_results_inner2['train_3_mean'].extend([np.mean(tr_acc)])
                            sk_results_inner2['val_2_mean'].extend([np.mean(val_acc)])

                            print('n_comp:', n_comp)
                            print('n train_3 fold 1:', data_train_3.shape[0])
                            print('Training 3 mean:', [np.mean(tr_acc)],
                                  'Validation 2 mean:', [np.mean(val_acc)])
                        # find best config for val 2
                        best_config_id = np.argmin(sk_results_inner2['val_2_mean'])
                        print('Best PCA config:', config_inner_2['n_comp'][best_config_id], '\n')
                        # fit optimum pipe

                        my_scaler = StandardScaler()
                        my_scaler.fit(data_train_2)
                        data_train_2 = my_scaler.transform(data_train_2)
                        data_val_1 = my_scaler.transform(data_val_1)

                        # Run PCA
                        my_pca = PCA_AE_Wrapper(n_components=config_inner_2['n_comp'][best_config_id])
                        my_pca.fit(data_train_2)
                        data_tr_2_pca = my_pca.transform(data_train_2)
                        data_val_1_pca = my_pca.transform(data_val_1)

                        # Run SVC
                        my_svc = SVC(kernel=current_kernel, C=c)
                        my_svc.fit(data_tr_2_pca, y_train_2)
                        svc_score_tr.append(my_svc.score(data_tr_2_pca, y_train_2))
                        svc_score_te.append(my_svc.score(data_val_1_pca, y_val_1))
                        print('Fit Optimum PCA Config and train with SVC')
                        print('n train 2:', data_train_2.shape[0])
                        print('n_comp:',config_inner_2['n_comp'][best_config_id])
                        print('SVC Train:', svc_score_tr[-1])
                        print('SVC Test:', svc_score_te[-1], '\n\n')
                        sk_results_inner1['train_2'].append(svc_score_tr[-1])
                        sk_results_inner1['val_1'].append(svc_score_te[-1])
                        fold_cnt += 1
                    sk_results_inner1['train_2_mean'].append(np.mean(svc_score_tr))
                    sk_results_inner1['val_1_mean'].append(np.mean(svc_score_te))


            print('\nNow find best config for SVC...')
            best_config_id_inner_1 = np.argmax(sk_results_inner1['val_1_mean'])
            print('Some test data:')
            print(data_test.shape)
            print(data_test[0:2,0:2])

            # fit optimum pipe
            my_scaler = StandardScaler()
            my_scaler.fit(data_train_1)
            data_train_1 = my_scaler.transform(data_train_1)
            data_test = my_scaler.transform(data_test)

            # Run PCA
            my_pca = PCA_AE_Wrapper(n_components=config_inner_2['n_comp'][best_config_id])
            my_pca.fit(data_train_1)
            data_tr_1_pca = my_pca.transform(data_train_1)
            data_test_pca = my_pca.transform(data_test)

            # Run SVC
            my_svc = SVC(kernel=config_inner_1['kernel'][best_config_id_inner_1],
                         C=config_inner_1['C'][best_config_id_inner_1])
            print('Best overall config:...')
            print('C = ',config_inner_1['C'][best_config_id_inner_1])
            print('kernel=', config_inner_1['kernel'][best_config_id_inner_1])
            print('pca_n_comp=', config_inner_2['n_comp'][best_config_id])
            print('n train 1:', data_train_1.shape[0])
            my_svc.fit(data_tr_1_pca, y_train_1)

            opt_tr_acc.append(my_svc.score(data_tr_1_pca, y_train_1))
            opt_test_acc.append(my_svc.score(data_test_pca, y_test))
            print('Train Acc:', opt_tr_acc[-1])
            print('Test Acc:', opt_test_acc[-1])

        print('\nCompare results of last iteration (outer cv)...')
        print('SkL  Train:', sk_results_inner1['train_2'])
        print('Pipe Train:', pipe_results['train'])
        print('SkL  Test: ', sk_results_inner1['val_1'])
        print('Pipe Test: ', pipe_results['test'])
        print('\nEval final performance:')
        print('Pipe final perf:', outer_pipe._test_performances['accuracy'])
        print('Sklearn final perf:', opt_test_acc)
        self.assertEqual(sk_results_inner1['train_2'], pipe_results['train'])
        self.assertEqual(sk_results_inner1['val_1'], pipe_results['test'])
        self.assertEqual(opt_test_acc, outer_pipe._test_performances['accuracy'])


if __name__ == '__main__':
    unittest.main()
