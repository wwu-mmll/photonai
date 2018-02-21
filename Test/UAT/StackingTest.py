import unittest, random, sklearn
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from photon_core.Framework.PhotonBase import Hyperpipe, \
    PipelineElement, PipelineStacking, SourceFilter
import numpy as np
from sklearn.svm import SVC

class CVTestsLocalSearchTrue(unittest.TestCase):
    thickness = None
    surface = None
    X = None
    y = None

    def setUp(self):
        random.seed(42)
        self.thickness = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_ThickAvg.csv').iloc[:, 1:]
        self.surface = pd.read_csv('../EnigmaTestFiles/CorticalMeasuresENIGMA_SurfAvg.csv').iloc[:, 1:]
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

        ##################################################################################
        # SKLEARN
        ##################################################################################


        for train1, test in cv_outer.split(self.X):
            X_train1 = self.X[train1]
            X_test = self.X[test]
            y_train1 = self.y[train1]
            y_test = self.y[test]

            sk_results = {'train_1': [], 'test': [], 'train_1_mean': [],
                          'test_mean': []}
            config_outer = {'C': [], 'kernel': []}

            for c_outer in svc_c:
                for kernel_outer in svc_kernel:
                    config_outer['C'].extend([c_outer])
                    config_outer['kernel'].extend([kernel_outer])

                    print('C:', c_outer, 'Kernel:', kernel_outer, '\n')

                    for train2, val1 in cv_inner.split(X_train1):
                        X_train2 = X_train1[train2]
                        X_val1 = X_train1[val1]
                        y_train2 = y_train1[train2]
                        y_val1 = y_train1[val1]

                        source_predictions = list()
                        for source in range(2):
                            results_source = {'C': list(), 'kernel': list(),
                                              'test_score': list(), 'test_predictions': list()}
                            for c_inner in svc_c:
                                for kernel_inner in svc_kernel:
                                    results_source['C'].append(c_inner)
                                    results_source['kernel'].append(kernel_inner)
                                    print('Source {} C:{} Kernel:{}\n'.format(source, c_outer, kernel_outer))

                                    results_source_folds = list()
                                    source_predictions_inner = list()
                                    all_val2_indices = list()
                                    for train3, val2 in cv_inner.split(X_train2):
                                        X_train3 = X_train2[train3][sources[source]]
                                        X_val2 = X_train2[val2][sources[source]]
                                        y_train3 = y_train2[train3]
                                        y_val2 = y_train2[val2]

                                        svc_source = SVC(kernel=kernel_inner, C=c_inner)
                                        svc_source.fit(X_train3, y_train3)
                                        source_predictions_inner.append(svc_source.predict(X_val2))
                                        all_val2_indices.append(val2)
                                        results_source_folds.append(svc_source.score(X_val2, y_val2))
                                    results_source['test_score'].append(np.mean(results_source_folds))
                                    # sort test predictions so that they are in the same order as y_val1 (or X_train2)
                                    results_source['test_predictions'].append(
                                        np.asarray(source_predictions_inner)[np.asarray(all_val2_indices)])
                            best_inner_config_id = np.argmax(results_source['test_score'])
                            best_inner_config = {'C': results_source['C'][best_inner_config_id],
                                                 'kernel': results_source['kernel'][best_inner_config_id]}
                            print('Optimum config for source {}: {}'.format(source, best_inner_config))
                            source_predictions.append(results_source['test_predictions'][best_inner_config_id])

                        # now we'll get our predictions as new data


