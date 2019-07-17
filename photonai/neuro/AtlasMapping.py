from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.BrainAtlas import BrainAtlas, AtlasLibrary
from photonai.base.PhotonBase import Hyperpipe
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
import pandas as pd
import os
import json


class AtlasMapper:
    def __init__(self):
        self.hyperpipes_to_fit = None
        self.folder = None
        self.hyperpipe_infos = None
        self.original_hyperpipe_name = None

    def generate_mappings(self, hyperpipe, folder):
        roi_list = list()
        target_element_name = ""
        self.original_hyperpipe_name = hyperpipe.name

        def found_brain_atlas(element):
            roi_list = element.base_element.rois
            if isinstance(roi_list, str):
                if roi_list == 'all':
                    atlas_obj = AtlasLibrary().get_atlas(element.base_element.atlas_name)
                    roi_list = atlas_obj.roi_list
                else:
                   roi_list = [roi_list]
            return roi_list

        # find brain atlas
        # first check preprocessing pipe
        if hyperpipe.preprocessing_pipe is not None:
            elements = hyperpipe.preprocessing_pipe.pipeline_elements
            preprocessing_flag = True
        else:
            elements = hyperpipe.pipeline_elements
            preprocessing_flag = False

        # then check usual pipeline_elements for a) NeuroModuleBranch -> check children and b) BrainAtlas directly
        for element in elements:
            if isinstance(element.base_element, NeuroModuleBranch):
                for neuro_element in element.base_element.pipeline_elements:
                    if isinstance(neuro_element.base_element, BrainAtlas):
                        target_element_name = neuro_element.name
                        roi_list = found_brain_atlas(neuro_element)

            elif isinstance(element.base_element, BrainAtlas):
                target_element_name = element.name
                roi_list = found_brain_atlas(element)

        hyperpipes_to_fit = dict()

        if len(roi_list) > 0:
            for roi_name in roi_list:
                copy_of_hyperpipe = hyperpipe.copy_me()
                new_pipe_name = copy_of_hyperpipe.name + '_Atlas_Mapper_' + roi_name
                copy_of_hyperpipe.name = new_pipe_name
                if preprocessing_flag:
                    copy_of_hyperpipe.preprocessing_pipe.set_params(**{target_element_name + "__rois": roi_name})
                else:
                    copy_of_hyperpipe.set_params(**{target_element_name + "__rois": roi_name})
                # mkdir could be needed?
                copy_of_hyperpipe.output_settings.project_folder = folder
                copy_of_hyperpipe.output_settings.overwrite_results = True
                copy_of_hyperpipe.output_settings.save_output = True
                hyperpipes_to_fit[roi_name] = copy_of_hyperpipe
        else:
            raise Exception("No Rois found...")
        self.hyperpipes_to_fit = hyperpipes_to_fit
        self.folder = folder

    def fit(self, X, y=None, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to fit. Did you call 'generate_mappings'?")

        hyperpipe_infos = dict()
        hyperpipe_results = dict()
        for roi_name, hyperpipe in self.hyperpipes_to_fit.items():
            hyperpipe.fit(X, y, **kwargs)
            hyperpipe_infos[roi_name] = {'hyperpipe_name': hyperpipe.name,
                                         'model_filename': hyperpipe.output_settings.pretrained_model_filename}
            hyperpipe_results[roi_name] = ResultsTreeHandler(hyperpipe.result_tree).get_performance_outer_folds()

        self.hyperpipe_infos = hyperpipe_infos
        with open(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_meta.json'), 'w') as fp:
            json.dump(self.hyperpipe_infos, fp)
        df = pd.DataFrame(hyperpipe_results)
        df.to_csv(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_results.csv'))

    def predict(self, X, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to predict. Did you remember to fit or load the Atlas Mapper?")

        predictions = dict()
        for roi, infos in self.hyperpipe_infos.items():
            predictions[roi] = self.hyperpipes_to_fit[roi].predict(X, **kwargs)

        return predictions

    def load_from_file(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError("Couldn't find atlas mapper meta file")

        with open(file, "r") as read_file:
            self.hyperpipe_infos = json.load(read_file)

        roi_models = dict()
        for roi_name, infos in self.hyperpipe_infos.items():
            roi_models[roi_name] = Hyperpipe.load_optimum_pipe(infos['model_filename'])
            self.hyperpipes_to_fit = roi_models
