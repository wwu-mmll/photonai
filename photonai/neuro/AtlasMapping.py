from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.BrainAtlas import BrainAtlas, AtlasLibrary
from photonai.base.PhotonBase import Hyperpipe, PhotonModelPersistor, PipelineElement
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
from photonai.photonlogger.Logger import Logger
import pandas as pd
import os
import json
from typing import Union
import joblib


class AtlasMapper:

    def __init__(self):
        self.folder = None
        self.neuro_element = None
        self.original_hyperpipe_name = None
        self.roi_list = None
        self.hyperpipe_infos = None
        self.hyperpipes_to_fit = None
        self.roi_indices = dict()

    def generate_mappings(self, hyperpipe: Hyperpipe, neuro_element: Union[NeuroModuleBranch, PipelineElement], folder: str):
        self.folder = folder
        self.neuro_element = neuro_element
        self.original_hyperpipe_name = hyperpipe.name
        self.roi_list = self._find_brain_atlas(self.neuro_element)
        self.hyperpipe_infos = None

        hyperpipes_to_fit = dict()

        if len(self.roi_list) > 0:
            for roi_index, roi_name in enumerate(self.roi_list):
                self.roi_indices[roi_name] = roi_index
                copy_of_hyperpipe = hyperpipe.copy_me()
                new_pipe_name = copy_of_hyperpipe.name + '_Atlas_Mapper_' + roi_name
                copy_of_hyperpipe.name = new_pipe_name
                copy_of_hyperpipe.output_settings.project_folder = folder
                copy_of_hyperpipe.output_settings.overwrite_results = True
                copy_of_hyperpipe.output_settings.save_output = True
                hyperpipes_to_fit[roi_name] = copy_of_hyperpipe
        else:
            raise Exception("No Rois found...")
        self.hyperpipes_to_fit = hyperpipes_to_fit

    def _find_brain_atlas(self, neuro_element):
        roi_list = list()
        if isinstance(neuro_element, NeuroModuleBranch):
            neuro_element.apply_groupwise = True
            for element in neuro_element.pipeline_elements:
                if isinstance(element.base_element, BrainAtlas):
                    element.base_element.collection_mode = 'list'
                    roi_list = self._find_rois(element)

        elif isinstance(neuro_element.base_element, BrainAtlas):
            neuro_element.base_element.collection_mode = 'list'
            roi_list = self._find_rois(neuro_element)
        return roi_list

    def _find_rois(self, element):
        roi_list = element.base_element.rois
        atlas_obj = AtlasLibrary().get_atlas(element.base_element.atlas_name)

        if isinstance(roi_list, str):
            if roi_list == 'all':
                roi_list = [roi.label for roi in atlas_obj.roi_list]
                roi_list.remove('Background')
            else:
                roi_list = [roi_list]
        elif isinstance(roi_list, list):
            valid_rois = list()

            for roi in roi_list:
                found_roi = False
                for valid_roi in atlas_obj.roi_list:
                    if roi == valid_roi.label:
                        valid_rois.append(valid_roi.label)
                        found_roi = True

                if not found_roi:
                    Logger().warn("{} is not a valid ROI for defined atlas. Skipping ROI.".format(roi))

            roi_list = valid_rois
        return roi_list

    def fit(self, X, y=None, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to fit. Did you call 'generate_mappings'?")

        # Get data from BrainAtlas first and save to .npz
        # ToDo: currently not supported for hyperparameters inside neurobranch
        self.neuro_element.fit(X)

        # save neuro branch to file
        joblib.dump(self.neuro_element, os.path.join(self.folder, 'neuro_element.pkl'), compress=1)

        # extract regions
        X_extracted, _, _ = self.neuro_element.transform(X)

        hyperpipe_infos = dict()
        hyperpipe_results = dict()

        for roi_name, hyperpipe in self.hyperpipes_to_fit.items():
            hyperpipe.fit(X_extracted[self.roi_indices[roi_name]], y, **kwargs)
            hyperpipe_infos[roi_name] = {'hyperpipe_name': hyperpipe.name,
                                         'model_filename': hyperpipe.output_settings.pretrained_model_filename,
                                         'roi_index': self.roi_indices[roi_name]}
            hyperpipe_results[roi_name] = ResultsTreeHandler(hyperpipe.result_tree).get_performance_outer_folds()

        self.hyperpipe_infos = hyperpipe_infos
        with open(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_meta.json'), 'w') as fp:
            json.dump(self.hyperpipe_infos, fp)
        df = pd.DataFrame(hyperpipe_results)
        df.to_csv(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_results.csv'))

    def predict(self, X, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to predict. Did you remember to fit or load the Atlas Mapper?")

        X_extracted, _, _ = self.neuro_element.transform(X)

        predictions = dict()
        for roi, infos in self.hyperpipe_infos.items():
            roi_index = infos['roi_index']
            predictions[roi] = self.hyperpipes_to_fit[roi].predict(X_extracted[roi_index], **kwargs)
        return predictions

    def load_from_file(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError("Couldn't find atlas mapper meta file")

        # load neuro branch
        self.folder = os.path.split(file)[0]
        self.neuro_element = joblib.load(os.path.join(self.folder, 'neuro_element.pkl'))

        with open(file, "r") as read_file:
            self.hyperpipe_infos = json.load(read_file)

        roi_models = dict()
        for roi_name, infos in self.hyperpipe_infos.items():
            roi_models[roi_name] = Hyperpipe.load_optimum_pipe(infos['model_filename'])
            self.hyperpipes_to_fit = roi_models
