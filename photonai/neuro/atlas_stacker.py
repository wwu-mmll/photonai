import pickle
from photonai.base import Hyperpipe, PipelineElement, Stack
from photonai.photonlogger.logger import logger

from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
import numpy as np

# TODO !!!
class RoiFilterElement(BaseEstimator):

    def __init__(self, roi_index):
        self.roi_index = roi_index

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return_data = X[self.roi_index]
        if isinstance(return_data, list):
            return_data = np.asarray(return_data)
        return return_data


class AtlasInfo:

    def __init__(self, atlas_name, roi_names, extraction_mode='mean', background_id=0, mask_threshold=None):
        self.atlas_name = atlas_name
        self.roi_names = roi_names
        self.extraction_mode = extraction_mode
        self.background_id = background_id
        self.mask_threshold = mask_threshold
        self.roi_names_runtime = []


class AtlasStacker(BaseEstimator):

    def __init__(self, atlas_info_object, hyperpipe_elements, best_config_metric=[], metrics=[]):
        # ToDo
        # - Stacker

        self.atlas_info_object = atlas_info_object
        self.atlas_name = self.atlas_info_object.atlas_name
        self.hyperpipe_elements = hyperpipe_elements
        self.pipeline_fusion = None
        self.best_config_metric = best_config_metric
        self.metrics = metrics
        # self.outer_pipe += pipeline_fusion

    def generate_hyperpipes(self):
        if self.atlas_info_object.roi_names_runtime:
            self.rois = self.atlas_info_object.roi_names_runtime
            #
            # self.outer_pipe = Hyperpipe(self.atlas_name + 'outer_pipe', optimizer='grid_search',
            #                        metrics=['accuracy'], hyperparameter_specific_config_cv_object=
            #                        ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
            #                        hyperparameter_search_cv_object=
                #                        ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                #                        eval_final_performance=True)

            inner_pipe_list = {}
            for i in range(len(self.rois)):
                tmp_inner_pipe = Hyperpipe(self.atlas_name + '_' + str(self.rois[i]), optimizer='grid_search',
                                           inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                                           eval_final_performance=False, verbose=logging.verbosity_level,
                                           best_config_metric=self.best_config_metric, metrics=self.metrics)

                # at first set a filter element

                roi_filter_element = RoiFilterElement(i)
                tmp_inner_pipe.filter_element = roi_filter_element

                # secondly add all other items
                for pipe_item in self.hyperpipe_elements:
                    tmp_inner_pipe += PipelineElement.create(pipe_item[0], pipe_item[1], **pipe_item[2])

                inner_pipe_list[self.rois[i]] = tmp_inner_pipe

            self.pipeline_fusion = Stack('multiple_source_pipes', inner_pipe_list.values(), voting=False)
        # Todo: else raise Error

    def fit(self, X, y=None):
        if not self.pipeline_fusion and not self.atlas_info_object.roi_names_runtime:
            raise BaseException('No ROIs could be received from Brain Atlas')

        elif not self.pipeline_fusion and self.atlas_info_object.roi_names_runtime:
            self.generate_hyperpipes()

        self.pipeline_fusion.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.pipeline_fusion.transform(X, y)


# if __name__ == '__main__':
#     from sklearn.model_selection import ShuffleSplit
#     from framework.PhotonBase import Hyperpipe, PipelineElement, Stack
#
#
#     pca_n_components = [5, 10]
#     svc_c = [0.1]
#     svc_kernel = ['linear', 'rbf']
#
#     # SET UP HYPERPIPE
#
#     outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
#                            metrics=['accuracy'], hyperparameter_specific_config_cv_object=
#                            ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
#                            hyperparameter_search_cv_object=
#                            ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
#                            eval_final_performance=True)
#
#
#     # Create pipe for 1st data source
#     pipe_source_1 = Hyperpipe('source_1', optimizer='grid_search',
#                               hyperparameter_specific_config_cv_object=
#                               ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
#                               eval_final_performance=False)
#     #    pipe_source_1.add(PipelineElement.create('SourceSplitter', {'column_indices': [np.arange(0, 10)]}))
#     pipe_source_1.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
#     pipe_source_1.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))
#
#     # Create pipe for 2nd data source
#     pipe_source_2 = Hyperpipe('source_1', optimizer='grid_search',
#                               hyperparameter_specific_config_cv_object=
#                               ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
#                               eval_final_performance=False)
#     pipe_source_2.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
#     pipe_source_2.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))
#
#     # stack sources
#     pipeline_fusion = Stack('multiple_source_pipes', [pipe_source_1, pipe_source_2], voting=False)
#     outer_pipe.add(pipeline_fusion)
#     outer_pipe.add(PipelineElement.create('svc'))
#     outer_pipe.fit(X, y)


    # def __getstate__(self): return self.__dict__
    #
    #
    # def __setstate__(self, d): self.__dict__.update(d)
    #
    # # Create pipe for second data source
    # pipe_source_2 = Hyperpipe('source_2', optimizer='grid_search',
    #                           hyperparameter_specific_config_cv_object=
    #                           ShuffleSplit(n_splits=1, test_size=0.2,
    #                                        random_state=3),
    #                           eval_final_performance=False)
    #
    # pipe_source_2.add(PipelineElement.create('SourceSplitter',
    #                                          {'column_indices': [np.arange(10, 20)]}))
    #
    # pipe_source_2.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
    # pipe_source_2.add(PipelineElement.create('svc', {'C': svc_c,
    #                                                  'kernel': svc_kernel}))
    # # Create pipe for third data source
    # pipe_source_3 = Hyperpipe('source_3', optimizer='grid_search',
    #                           hyperparameter_specific_config_cv_object=
    #                           ShuffleSplit(n_splits=1, test_size=0.2,
    #                                        random_state=3),
    #                           eval_final_performance=False)
    #
    # pipe_source_3.add(PipelineElement.create('SourceSplitter', {
    #     'column_indices': [np.arange(20, 30)]}))
    # pipe_source_3.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
    # pipe_source_3.add(PipelineElement.create('svc', {'C': svc_c,
    #                                                  'kernel': svc_kernel}))
    #
    # # pipeline_fusion = Stack('multiple_source_pipes',[pipe_source_1, pipe_source_2, pipe_source_3], voting=False)
    # pipeline_fusion = Stack('multiple_source_pipes',
    #                                    [pipe_source_1, pipe_source_2, pipe_source_3])
    #
    # outer_pipe.add(pipeline_fusion)
    # # outer_pipe.add(PipelineElement.create('svc', {'C': svc_c_2, 'kernel': svc_kernel}))
    # # outer_pipe.add(PipelineElement.create('knn',{'n_neighbors':[15]}))
    # outer_pipe.add(PipelineElement.create('kdnn', {'target_dimension': [2], 'nb_epoch': [10]}))
    #
    # # START HYPERPARAMETER SEARCH
    # outer_pipe.fit(self.__X, self.__y)
    # print(outer_pipe.test_performances)
    # pipe_results = {'train': [], 'test': []}
    # for i in range(int(len(outer_pipe.performance_history_list) / 2)):
    #     pipe_results['train'].extend(
    #         outer_pipe.performance_history_list[i]['accuracy_folds']['train'])
    #     pipe_results['test'].extend(
    #         outer_pipe.performance_history_list[i]['accuracy_folds']['test'])

    # print(outer_pipe.test_performances['accuracy'])
