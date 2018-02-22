import unittest, random, sklearn
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from photon_core.Framework.PhotonBase import Hyperpipe, \
    PipelineElement, PipelineStacking, SourceFilter
import numpy as np
from sklearn.svm import SVC

class StackingTest(unittest.TestCase):
    thickness = None
    surface = None
    X = None
    y = None

    def setUp(self):
        random.seed(49)
        self.thickness = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv').iloc[:, 1:20]
        self.surface = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_SurfAvg.csv').iloc[:, 1:20]
        y = pd.read_csv('../EnigmaTestFiles/Covariates.csv')['Sex']
        thick_mean = np.mean(self.thickness)
        surf_mean = np.mean(self.surface)
        for i in range(self.surface.shape[0]):
            for k in range(self.surface.shape[1]):
                if str(self.surface.iloc[i, k]) == 'nan':
                    self.surface.iloc[i, k] = surf_mean[k]
            for k in range(self.thickness.shape[1]):
                if str(self.thickness.iloc[i, k]) == 'nan':
                    self.thickness.iloc[i, k] = thick_mean[k]

        # Concatenate
        X = np.concatenate((self.surface, self.thickness), axis=1)
        self.X = np.asarray(X, order='C', dtype='float64')
        self.y = np.asarray(y, dtype='float64')


    def testStacking(self):
        svc_c = [.1, 1]
        svc_kernel = ['linear']

        cv_outer = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
        # cv_inner = ShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
        #cv_outer = KFold(n_splits=3, random_state=3)
        cv_inner = KFold(n_splits=3, random_state=3)

        sources = [np.arange(0, self.surface.shape[1]),np.arange(self.surface.shape[1], self.surface.shape[1] + self.thickness.shape[1])]

        ##################################################################################
        # SET UP HYPERPIPES
        ##################################################################################

        # surface pipe
        surface_pipe = Hyperpipe('surface_pipe', optimizer='grid_search',
                                 metrics=['accuracy'],
                                 inner_cv=cv_inner, verbose=1)

        surface_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})
        # use source filter to select data for stacked hyperpipes
        surface_pipe.filter_element = SourceFilter(sources[0])

        # thickness pipe
        thickness_pipe = Hyperpipe('thickness_pipe', optimizer='grid_search',
                                   metrics=['accuracy'],
                                   inner_cv=cv_inner, verbose=1)

        thickness_pipe += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})
        # use source filter to select data for stacked hyperpipes
        thickness_pipe.filter_element = SourceFilter(sources[1])

        # Mother Pipe
        mother = Hyperpipe('mother', optimizer='grid_search',
                           metrics=['accuracy'],
                           inner_cv=cv_inner,
                           outer_cv=cv_outer,
                           eval_final_performance=True, verbose=1)

        mother += PipelineStacking('multiple_sources', [surface_pipe, thickness_pipe], voting=False)
        mother += PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel})

        mother.fit(self.X, self.y)
        final_score_photon = mother.result_tree.get_best_config_performance_test_set(0).metrics['accuracy']

        ##################################################################################
        # SKLEARN
        ##################################################################################

        for train1, test in cv_outer.split(self.X):
            X_train1 = self.X[train1]
            X_test = self.X[test]
            y_train1 = self.y[train1]
            y_test = self.y[test]

            results_outer = {'C': [], 'kernel': [], 'val1_score': []}
            done_source_optimization = False
            for c_outer in svc_c:
                for kernel_outer in svc_kernel:
                    results_outer['C'].extend([c_outer])
                    results_outer['kernel'].extend([kernel_outer])

                    print('C Outer:', c_outer, 'Kernel Outer:', kernel_outer, '\n')
                    results_val1 = []
                    for train2, val1 in cv_inner.split(X_train1):
                        X_train2 = X_train1[train2]
                        X_val1 = X_train1[val1]
                        y_train2 = y_train1[train2]
                        y_val1 = y_train1[val1]

                        if done_source_optimization is not True:
                            source_predictions_train2 = list()
                            source_predictions_val1 = list()
                            best_inner_config = []

                            for source in range(2):
                                results_source = {'C': list(), 'kernel': list(),
                                                  'test_score': list(), 'test_predictions': list()}

                                for c_inner in svc_c:
                                    for kernel_inner in svc_kernel:
                                        results_source['C'].append(c_inner)
                                        results_source['kernel'].append(kernel_inner)
                                        print('Source {} C:{} Kernel:{}\n'.format(source, c_inner, kernel_inner))

                                        results_source_folds = list()
                                        for train3, val2 in cv_inner.split(X_train2):
                                            X_train3 = X_train2[train3][:, sources[source]]
                                            X_val2 = X_train2[val2][:, sources[source]]
                                            y_train3 = y_train2[train3]
                                            y_val2 = y_train2[val2]

                                            svc_source = SVC(kernel=kernel_inner, C=c_inner)
                                            svc_source.fit(X_train3, y_train3)
                                            results_source_folds.append(svc_source.score(X_val2, y_val2))
                                        results_source['test_score'].append(np.mean(results_source_folds))

                                best_inner_config_id = np.argmax(results_source['test_score'])
                                best_inner_config.append({'C': results_source['C'][best_inner_config_id],
                                                          'kernel': results_source['kernel'][best_inner_config_id]})
                                print('Optimum config for source {}: {}'.format(source, best_inner_config[-1]))
                                print('Now fitting optimum source pipe...')
                                svc_source_opt = SVC(C=best_inner_config[-1]['C'],
                                                     kernel=best_inner_config[-1]['kernel'])
                                svc_source_opt.fit(X_train2[:, sources[source]], y_train2)
                                source_predictions_train2.append(svc_source_opt.predict(X_train2[:, sources[source]]))
                                source_predictions_val1.append(svc_source_opt.predict(X_val1[:, sources[source]]))
                            done_source_optimization = True
                        else:
                            print('Skipping optimization of sources')
                        print('Now fit 2nd level classifier with C={} and kernel={}'.format(c_outer, kernel_outer))
                        svc_meta = SVC(C=c_outer, kernel=kernel_outer)
                        svc_meta.fit(np.transpose(np.asarray(source_predictions_train2)), y_train2)
                        results_val1.append(svc_meta.score(np.transpose(np.asarray(source_predictions_val1)), y_val1))
                    results_outer['val1_score'].append(np.mean(results_val1))
            best_outer_config_id = np.argmax(results_outer['val1_score'])
            best_outer_config = {'C': results_outer['C'][best_outer_config_id],
                                 'kernel': results_outer['kernel'][best_outer_config_id]}
            print('Optimum config for meta classifier: {}'.format(best_outer_config))
            print('Now fitting optimum meta pipe...')
            print('...with source config for source 1: {} and source 2: {}'.format(best_inner_config[0],
                                                                                   best_inner_config[1]))
            svc_meta_opt = SVC(C=best_outer_config['C'], kernel=best_outer_config['kernel'])
            svc_source_1_opt = SVC(C=best_inner_config[0]['C'], kernel=best_inner_config[0]['kernel'])
            svc_source_2_opt = SVC(C=best_inner_config[1]['C'], kernel=best_inner_config[1]['kernel'])
            svc_source_1_opt.fit(X_train1[:, sources[0]], y_train1)
            svc_source_2_opt.fit(X_train1[:, sources[1]], y_train1)
            pred_source1_train1 = svc_source_1_opt.predict(X_train1[:, sources[0]])
            pred_source2_train1 = svc_source_2_opt.predict(X_train1[:, sources[1]])
            svc_meta_opt.fit(np.transpose(np.asarray([pred_source1_train1, pred_source2_train1])), y_train1)

            # get test performance
            pred_source1_test = svc_source_1_opt.predict(X_test[:, sources[0]])
            pred_source2_test = svc_source_2_opt.predict(X_test[:, sources[1]])
            final_score = svc_meta_opt.score(np.transpose(np.asarray([pred_source1_test, pred_source2_test])), y_test)
            print('Final test performance: {}'.format(final_score))


        self.assertEqual(final_score, final_score_photon)


if __name__ == '__main__':
    unittest.main()
