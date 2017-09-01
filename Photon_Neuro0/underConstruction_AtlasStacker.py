import pickle
from sklearn.base import BaseEstimator

class AtlasStacker(BaseEstimator):
    def __init__(self, atlas_name=None, extract_mode='mean', whichROIs='all', background_id=0):
        # ToDo
        # - Stacker

        # get info about available atlases
        ATLAS_DICT, atlas_dir = BrainAtlas._getAtlasDict()

    def fit(self):
        pass

    def transform(self, X, y=None):
        extract_mode = self.extract_mode
        whichROIs = self.whichROIs
        background_id = self.background_id
        self.gotData = True

        # get ROI infos
        rois = self._getROIs(whichROIs=whichROIs, background_id=background_id)

        # Grab masker and apply to structural data for each ROI
        from nilearn.input_data import NiftiMasker
        roi_data = []
        if extract_mode == 'box': self.box_shape = []

        for roi in rois:
            masker = NiftiMasker(mask_img=roi)
            single_roi = masker.fit_transform(X)
            if extract_mode == 'vec':
                roi_data.append(single_roi)
            elif extract_mode == 'mean':
                tmp = []
                for sample_ind in range(len(single_roi)):
                    tmp.append(np.mean(single_roi[sample_ind]))
                roi_data.append(tmp)
            elif extract_mode == 'box':
                tmp = []
                tmp, bshape = self._getBox(X, roi)
                roi_data.append(tmp)
                self.box_shape.append(bshape)
            else:
                # any function which can work on a vector passed as a string
                tmp = []
                # find something safer than eval!
                any_func = lambda ex, data, opt_args=None: eval(ex)(data, opt_args)
                expr = 'np.std'
                for sample_ind in range(0, len(single_roi)):
                    t = any_func(expr, single_roi[sample_ind])
                    tmp.append(t)
                roi_data.append(tmp)

        return roi_data

if __name__ == '__main__':
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    from Framework.PhotonBase import Hyperpipe, PipelineElement, PipelineStacking


    pca_n_components = [5, 10]
    svc_c = [0.1]
    svc_c_2 = [1]
    # svc_kernel = ['rbf']
    svc_kernel = ['linear']

    # SET UP HYPERPIPE

    outer_pipe = Hyperpipe('outer_pipe', optimizer='grid_search',
                           metrics=['accuracy'], hyperparameter_specific_config_cv_object=
                           ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                           hyperparameter_search_cv_object=
                           ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                           eval_final_performance=True)


    # Create pipe for first data source
    pipe_source_1 = Hyperpipe('source_1', optimizer='grid_search',
                              hyperparameter_specific_config_cv_object=
                              ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                              eval_final_performance=False)

    # from copy import deepcopy
    # pipe_source_2 = deepcopy(pipe_source_1)

    #    pipe_source_1.add(PipelineElement.create('SourceSplitter', {'column_indices': [np.arange(0, 10)]}))
    pipe_source_1.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
    pipe_source_1.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))

    # Create pipe for first data source
    pipe_source_2 = Hyperpipe('source_1', optimizer='grid_search',
                              hyperparameter_specific_config_cv_object=
                              ShuffleSplit(n_splits=1, test_size=0.2, random_state=3),
                              eval_final_performance=False)

    #    pipe_source_1.add(PipelineElement.create('SourceSplitter', {'column_indices': [np.arange(0, 10)]}))
    pipe_source_2.add(PipelineElement.create('pca', {'n_components': pca_n_components}))
    pipe_source_2.add(PipelineElement.create('svc', {'C': svc_c, 'kernel': svc_kernel}))
    print('1')


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
    # # pipeline_fusion = PipelineStacking('multiple_source_pipes',[pipe_source_1, pipe_source_2, pipe_source_3], voting=False)
    # pipeline_fusion = PipelineStacking('multiple_source_pipes',
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

    print(outer_pipe.test_performances['accuracy'])
