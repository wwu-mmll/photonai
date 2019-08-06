from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.BrainAtlas import BrainAtlas, AtlasLibrary
from photonai.base.PhotonBase import Hyperpipe, PhotonModelPersistor
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
from photonai.photonlogger.Logger import Logger
import pandas as pd
import os
import json


class AtlasMapper:
    def __init__(self):
        self.hyperpipes_to_fit = None
        self.folder = None
        self.hyperpipe_infos = None
        self.original_hyperpipe_name = None
        self.neuro_branch = None

    def generate_mappings(self, hyperpipe, folder):
        roi_list = list()
        self.original_hyperpipe_name = hyperpipe.name

        def find_brain_atlas(element):
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

        # find brain atlas
        # first check preprocessing pipe
        if hyperpipe.preprocessing_pipe is not None:
            elements = hyperpipe.preprocessing_pipe.pipeline_elements
            preprocessing_flag = True
        else:
            elements = hyperpipe.pipeline_elements
            preprocessing_flag = False

        # then check usual pipeline_elements for a) NeuroModuleBranch -> check children and b) BrainAtlas directly
        for i, element in enumerate(elements):
            if isinstance(element.base_element, NeuroModuleBranch):
                self.neuro_branch = element.base_element.copy_me()
                for neuro_element in element.base_element.pipeline_elements:
                    if isinstance(neuro_element.base_element, BrainAtlas):
                        # target_element_name = neuro_element.name
                        roi_list = find_brain_atlas(neuro_element)
                # delete neurobranch from hyperpipe
                del elements[i]
            elif isinstance(element.base_element, BrainAtlas):
                self.neuro_branch = element.copy_me()
                # target_element_name = element.name
                roi_list = find_brain_atlas(element)
                # delete brain atlas
                del elements [i]

        # in case it's a preprocessing pipe that is now empty, delete it completely
        if preprocessing_flag:
            if not hyperpipe.preprocessing_pipe.pipeline_elements:
                del hyperpipe.preprocessing_pipe

        hyperpipes_to_fit = dict()

        if len(roi_list) > 0:
            for roi_name in roi_list:
                roi_name = roi_name
                copy_of_hyperpipe = hyperpipe.copy_me()
                new_pipe_name = copy_of_hyperpipe.name + '_Atlas_Mapper_' + roi_name
                copy_of_hyperpipe.name = new_pipe_name
                # if preprocessing_flag:
                #     copy_of_hyperpipe.preprocessing_pipe.set_params(**{target_element_name + "__rois": roi_name})
                # else:
                #     copy_of_hyperpipe.set_params(**{target_element_name + "__rois": roi_name})
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

        # Get data from BrainAtlas first and save to .npz
        # ToDo: currently not supported for hyperparameters inside neurobranch
        self.neuro_branch.fit(X)
        # save neuro branch to file
        PhotonModelPersistor.save_optimum_pipe(self.neuro_branch, os.path.join(self.folder, 'neuro_branch.photon'))
        X_extracted, _, _ = self.neuro_branch.transform(X)

        hyperpipe_infos = dict()
        hyperpipe_results = dict()
        roi_counter = 0

        for roi_name, hyperpipe in self.hyperpipes_to_fit.items():
            hyperpipe.fit(X_extracted[roi_counter], y, **kwargs)
            hyperpipe_infos[roi_name] = {'hyperpipe_name': hyperpipe.name,
                                         'model_filename': hyperpipe.output_settings.pretrained_model_filename}
            hyperpipe_results[roi_name] = ResultsTreeHandler(hyperpipe.result_tree).get_performance_outer_folds()
            roi_counter += 1

        self.hyperpipe_infos = hyperpipe_infos
        with open(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_meta.json'), 'w') as fp:
            json.dump(self.hyperpipe_infos, fp)
        df = pd.DataFrame(hyperpipe_results)
        df.to_csv(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_results.csv'))

    def predict(self, X, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to predict. Did you remember to fit or load the Atlas Mapper?")

        X_extracted = self.neuro_branch.transform(X)

        predictions = dict()
        roi_counter = 0
        for roi, infos in self.hyperpipe_infos.items():
            predictions[roi] = self.hyperpipes_to_fit[roi].predict(X_extracted[roi_counter], **kwargs)
            roi_counter += 1
        return predictions

    def load_from_file(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError("Couldn't find atlas mapper meta file")

        # load neuro branch
        self.folder = os.path.split(file)[0]
        self.neuro_branch = PhotonModelPersistor.load_optimum_pipe(os.path.join(self.folder, 'neuro_branch.photon'))

        with open(file, "r") as read_file:
            self.hyperpipe_infos = json.load(read_file)

        roi_models = dict()
        for roi_name, infos in self.hyperpipe_infos.items():
            roi_models[roi_name] = Hyperpipe.load_optimum_pipe(infos['model_filename'])
            self.hyperpipes_to_fit = roi_models
