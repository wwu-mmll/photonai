# Case C2: Multi-Source Nested Hyperparameter Optimization with Stacking and Local Optimization


import random
import unittest

import numpy as np
from sklearn.model_selection import ShuffleSplit

from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineStacking

np.random.seed(3)


class CVTestCaseC2(unittest.TestCase):
    __X = None
    __y = None

    def setUp(self):
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        self.__X = dataset.data
        self.__y = dataset.target
        self.__X1 = self.__X[:,0:10]
        self.__X2 = self.__X[:,10:20]
        self.__X3 = self.__X[:,20:30]
        random.seed(42)

    def testCaseC2(self):
        pca_n_components = [5, 10]
        svc_c = [0.1]
        svc_c_2 = [1]
        #svc_kernel = ['rbf']
        svc_kernel = ['linear']

        # SET UP HYPERPIPE

        outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
                               metrics=['accuracy'], inner_cv=
                               ShuffleSplit(n_splits=1,test_size=0.2, random_state=3),
                               outer_cv=
                               ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                               eval_final_performance=True)

        # Create pipe for first data source
        pipe_source_1 = Hyperpipe('source_1', optimizer='grid_search',
                                  inner_cv=
                               ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                                  eval_final_performance=False)

        pipe_source_1.add(PipelineElement.create('SourceSplitter',{'column_indices': [np.arange(0,10)]}))
        pipe_source_1.add(PipelineElement.create('pca',{'n_components': pca_n_components}))
        pipe_source_1.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))

        # Create pipe for second data source
        pipe_source_2 = Hyperpipe('source_2', optimizer='grid_search',
                                  inner_cv=
                                 ShuffleSplit(n_splits=1, test_size=0.2,
                                              random_state=3),
                                  eval_final_performance=False)

        pipe_source_2.add(PipelineElement.create('SourceSplitter',
                                                 {'column_indices': [np.arange(10, 20)]}))

        pipe_source_2.add(PipelineElement.create('pca',{'n_components': pca_n_components}))
        pipe_source_2.add(PipelineElement.create('svc', {'C': svc_c,
                                                        'kernel': svc_kernel}))
        # Create pipe for third data source
        pipe_source_3 = Hyperpipe('source_3', optimizer='grid_search',
                                  inner_cv=
                                 ShuffleSplit(n_splits=1, test_size=0.2,
                                              random_state=3),
                                  eval_final_performance=False)

        pipe_source_3.add(PipelineElement.create('SourceSplitter', {
            'column_indices': [np.arange(20, 30)]}))
        pipe_source_3.add(PipelineElement.create('pca',{'n_components': pca_n_components}))
        pipe_source_3.add(PipelineElement.create('svc', {'C': svc_c,
                                                        'kernel': svc_kernel}))


        # pipeline_fusion = PipelineStacking('multiple_source_pipes',[pipe_source_1, pipe_source_2, pipe_source_3], voting=False)
        pipeline_fusion = PipelineStacking('multiple_source_pipes',
                                           [pipe_source_1, pipe_source_2, pipe_source_3])

        outer_pipe.add(pipeline_fusion)
        #outer_pipe.add(PipelineElement.create('svc', {'C': svc_c_2, 'kernel': svc_kernel}))
        #outer_pipe.add(PipelineElement.create('knn',{'n_neighbors':[15]}))
        outer_pipe.add(PipelineElement.create('kdnn', {'target_dimension': [2],'nb_epoch':[10]}))

        # START HYPERPARAMETER SEARCH
        outer_pipe.fit(self.__X, self.__y)
        print(outer_pipe._test_performances)
        pipe_results = {'train': [], 'test': []}
        for i in range(int(len(outer_pipe._performance_history_list)/2)):
            pipe_results['train'].extend(
                outer_pipe._performance_history_list[i]['accuracy_folds']['train'])
            pipe_results['test'].extend(
                outer_pipe._performance_history_list[i]['accuracy_folds']['test'])

        print(outer_pipe._test_performances['accuracy'])




        # print('\n\n')
        # print('Running sklearn version...\n')
        # #cv_outer = KFold(n_splits=3, random_state=3)
        # cv_outer = ShuffleSplit(n_splits=1,test_size=0.2, random_state=3)
        # cv_inner_1 = ShuffleSplit(n_splits=1,test_size=0.2, random_state=3)
        # cv_inner_2 = ShuffleSplit(n_splits=1,test_size=0.2, random_state=3)
        # # cv_inner_1 = KFold(n_splits=2, random_state=3)
        # # cv_inner_2 = KFold(n_splits=2, random_state=3)
        #
        # opt_tr_acc = []
        # opt_test_acc = []
        #
        # for train_1, test in cv_outer.split(self.__X):
        #     data_train_1 = self.__X[train_1]
        #     data_test = self.__X[test]
        #     y_train_1 = self.__y[train_1]
        #     y_test = self.__y[test]
        #     sk_results = {'train_1': [], 'test': [], 'train_1_mean': [],
        #                   'test_mean': []}
        #     config_inner_1 = {'C':[], 'kernel':[]}
        #     sk_results_inner1 = {'train_2': [],
        #                          'val_1': [],
        #                          'train_2_mean': [],
        #                          'val_1_mean': []}
        #     print('Outer Split')
        #     print('n train_1:', data_train_1.shape[0], '\n')
        #
        #
        #     for c in svc_c:
        #         for current_kernel in svc_kernel:
        #             config_inner_1['C'].extend([c])
        #             config_inner_1['kernel'].extend([current_kernel])
        #
        #             print('C:', c, 'Kernel:', current_kernel, '\n')
        #             svc_score_tr = []
        #             svc_score_te = []
        #             fold_cnt = 1
        #             for train_2, val_1 in cv_inner_1.split(
        #                     data_train_1):
        #                 print('\n\nSklearn Outer Pipe FoldMetrics', fold_cnt)
        #
        #                 data_train_2 = data_train_1[train_2]
        #                 data_val_1 = data_train_1[val_1]
        #                 y_train_2 = y_train_1[train_2]
        #                 y_val_1 = y_train_1[val_1]
        #                 print('n train_2:', data_train_2.shape[0], '\n')
        #
        #
        #                 config_inner_2 = {'n_comp': []}
        #                 print('Sklearn PCA Pipe')
        #                 for n_comp in pca_n_components:
        #                     config_inner_2['n_comp'].extend([n_comp])
        #
        #                     tr_acc = []
        #                     val_acc = []
        #                     sk_results_inner2 = {'train_3': [], 'val_2': [],
        #                                   'train_3_mean': [],
        #                                   'val_2_mean': []}
        #                     # print('Some training data:',
        #                     #       data_train_2[0:2, 0:2])
        #                     for train_3, val_2 in cv_inner_2.split(
        #                             data_train_2):
        #
        #                         data_train_3 = data_train_2[train_3]
        #
        #                         data_val_2 = data_train_2[val_2]
        #                         y_train_3 = y_train_2[train_3]
        #                         y_val_2 = y_train_2[val_2]
        #
        #                         my_pca = PCA()
        #                         my_pca.fit(data_train_3)
        #                         data_train_3 = my_pca.transform(data_train_3)
        #
        #                         # Run Source SVM
        #                         my_pca = SVM(n_components=n_comp)
        #                         my_pca.fit(data_train_3)
        #                         data_tr_3_pca_inv = my_pca.transform(data_train_3)
        #                         data_val_2_pca_inv = my_pca.transform(data_val_2)
        #
        #                         mae_tr = my_pca.score(data_train_3)
        #                         mae_te = my_pca.score(data_val_2)
        #
        #                         tr_acc.append(mae_tr)
        #                         val_acc.append(mae_te)
        #
        #                     sk_results_inner2['train_3'].extend(tr_acc)
        #                     sk_results_inner2['val_2'].extend(val_acc)
        #                     sk_results_inner2['train_3_mean'].extend([np.mean(tr_acc)])
        #                     sk_results_inner2['val_2_mean'].extend([np.mean(val_acc)])
        #
        #                     print('n_comp:', n_comp)
        #                     print('n train_3 fold 1:', data_train_3.shape[0])
        #                     print('Training 3 mean:', [np.mean(tr_acc)],
        #                           'validation 2 mean:', [np.mean(val_acc)])
        #                 # find best config for val 2
        #                 best_config_id = np.argmax(sk_results_inner2['val_2_mean'])
        #                 print('Best PCA config:', config_inner_2['n_comp'][best_config_id], '\n')
        #                 # fit optimum pipe
        #
        #                 my_scaler = StandardScaler()
        #                 my_scaler.fit(data_train_2)
        #                 data_train_2 = my_scaler.transform(data_train_2)
        #                 data_val_1 = my_scaler.transform(data_val_1)
        #
        #                 # Run PCA
        #                 my_pca = PCA_AE_Wrapper(n_components=config_inner_2['n_comp'][best_config_id])
        #                 my_pca.fit(data_train_2)
        #                 data_tr_2_pred = SVM_trans.transform(data_train_2)
        #                 data_val_1_pred = SVM_trans.transform(data_val_1)
        #
        #                 # Run SVC
        #                 my_svc = SVC(kernel=current_kernel, C=c)
        #                 my_svc.fit(data_tr_2_pred, y_train_2)
        #                 svc_score_tr.append(my_svc.score(data_tr_2_pca, y_train_2))
        #                 svc_score_te.append(my_svc.score(data_val_1_pca, y_val_1))
        #                 print('Fit Optimum PCA Config and train with SVC')
        #                 print('n train 2:', data_train_2.shape[0])
        #                 print('n_comp:',config_inner_2['n_comp'][best_config_id])
        #                 print('SVC Train:', svc_score_tr[-1])
        #                 print('SVC Test:', svc_score_te[-1], '\n\n')
        #                 sk_results_inner1['train_2'].append(svc_score_tr[-1])
        #                 sk_results_inner1['val_1'].append(svc_score_te[-1])
        #                 fold_cnt += 1
        #             sk_results_inner1['train_2_mean'].append(np.mean(svc_score_tr))
        #             sk_results_inner1['val_1_mean'].append(np.mean(svc_score_te))
        #
        #
        #     print('\nNow find best config for SVC...')
        #     best_config_id_inner_1 = np.argmax(sk_results_inner1['val_1_mean'])
        #     print('Some test data:')
        #     print(data_test.shape)
        #     print(data_test[0:2,0:2])
        #
        #     # fit optimum pipe
        #     my_scaler = StandardScaler()
        #     my_scaler.fit(data_train_1)
        #     data_train_1 = my_scaler.transform(data_train_1)
        #     data_test = my_scaler.transform(data_test)
        #
        #     # Run PCA
        #     my_pca = PCA_AE_Wrapper(n_components=config_inner_2['n_comp'][best_config_id])
        #     my_pca.fit(data_train_1)
        #     data_tr_1_pca = my_pca.transform(data_train_1)
        #     data_test_pca = my_pca.transform(data_test)
        #
        #     # Run SVC
        #     my_svc = SVC(kernel=config_inner_1['kernel'][best_config_id_inner_1],
        #                  C=config_inner_1['C'][best_config_id_inner_1])
        #     print('Best overall config:...')
        #     print('C = ',config_inner_1['C'][best_config_id_inner_1])
        #     print('kernel=', config_inner_1['kernel'][best_config_id_inner_1])
        #     print('pca_n_comp=', config_inner_2['n_comp'][best_config_id])
        #     print('n train 1:', data_train_1.shape[0])
        #     my_svc.fit(data_tr_1_pca, y_train_1)
        #
        #     opt_tr_acc.append(my_svc.score(data_tr_1_pca, y_train_1))
        #     opt_test_acc.append(my_svc.score(data_test_pca, y_test))
        #     print('Train Acc:', opt_tr_acc[-1])
        #     print('Test Acc:', opt_test_acc[-1])
        #
        # print('\nCompare results of last iteration (outer cv)...')
        # print('SkL  Train:', sk_results_inner1['train_2'])
        # print('Pipe Train:', pipe_results['train'])
        # print('SkL  Test: ', sk_results_inner1['val_1'])
        # print('Pipe Test: ', pipe_results['test'])
        # print('\nEval final performance:')
        # print('Pipe final perf:', outer_pipe.test_performances['accuracy'])
        # print('Sklearn final perf:', opt_test_acc)
        # # self.assertEqual(sk_results_inner1['train_2'], pipe_results['train'])
        # # self.assertEqual(sk_results_inner1['val_1'], pipe_results['test'])
        # self.assertEqual(opt_test_acc, outer_pipe.test_performances['accuracy'])


if __name__ == '__main__':
    unittest.main()
