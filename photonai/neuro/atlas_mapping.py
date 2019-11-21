import json
import os
from glob import glob
from typing import Union

import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from nilearn import datasets, surface, plotting

from photonai.base import PipelineElement
from photonai.base.hyperpipe import Hyperpipe
from photonai.neuro.brain_atlas import BrainAtlas, AtlasLibrary
from photonai.neuro.neuro_branch import NeuroBranch
from photonai.photonlogger.logger import logger

from photonai.processing import ResultsHandler


class AtlasMapper:

    def __init__(self, create_surface_plots: bool = False):
        self.folder = None
        self.neuro_element = None
        self.original_hyperpipe_name = None
        self.roi_list = None
        self.atlas = None
        self.hyperpipe_infos = None
        self.hyperpipes_to_fit = None
        self.roi_indices = dict()
        self.best_config_metric = None
        self.create_surface_plots = create_surface_plots

    def generate_mappings(self, hyperpipe: Hyperpipe, neuro_element: Union[NeuroBranch, PipelineElement], folder: str):
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.neuro_element = neuro_element
        self.original_hyperpipe_name = hyperpipe.name
        self.roi_list, self.atlas = self._find_brain_atlas(self.neuro_element)
        self.verbosity = hyperpipe.verbosity
        self.hyperpipe_infos = None
        self.best_config_metric = hyperpipe.optimization.best_config_metric

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
        atlas_obj = list()
        if isinstance(neuro_element, NeuroBranch):
            for element in neuro_element.elements:
                if isinstance(element.base_element, BrainAtlas):
                    element.base_element.collection_mode = 'list'
                    roi_list, atlas_obj = self._find_rois(element)

        elif isinstance(neuro_element.base_element, BrainAtlas):
            neuro_element.base_element.collection_mode = 'list'
            roi_list, atlas_obj = self._find_rois(neuro_element)
        return roi_list, atlas_obj

    @staticmethod
    def _find_rois(element):
        roi_list = element.base_element.rois
        atlas_obj = AtlasLibrary().get_atlas(element.base_element.atlas_name)
        roi_objects = BrainAtlas._get_rois(atlas_obj, roi_list)
        return [roi.label for roi in roi_objects], atlas_obj

    def fit(self, X, y=None, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to fit. Did you call 'generate_mappings'?")

        # Get data from BrainAtlas first and save to .npz
        # ToDo: currently not supported for hyperparameters inside neurobranch
        self.neuro_element.fit(X)

        # extract regions
        X_extracted, _, _ = self.neuro_element.transform(X)
        X_extracted = self._reshape_roi_data(X_extracted)

        # save neuro branch to file
        joblib.dump(self.neuro_element, os.path.join(self.folder, 'neuro_element.pkl'), compress=1)

        hyperpipe_infos = dict()
        hyperpipe_results = dict()

        for roi_name, hyperpipe in self.hyperpipes_to_fit.items():
            hyperpipe.verbosity = self.verbosity
            hyperpipe.fit(X_extracted[self.roi_indices[roi_name]], y, **kwargs)
            hyperpipe_infos[roi_name] = {'hyperpipe_name': hyperpipe.name,
                                         'model_filename': os.path.join(os.path.basename(hyperpipe.output_settings.results_folder),
                                                                        'photon_best_model.photon'),
                                         'roi_index': self.roi_indices[roi_name]}
            hyperpipe_results[roi_name] = ResultsHandler(hyperpipe.results).get_performance_outer_folds()

        self.hyperpipe_infos = hyperpipe_infos

        # write results
        with open(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_meta.json'), 'w') as fp:
            json.dump(self.hyperpipe_infos, fp)
        df = pd.DataFrame(hyperpipe_results)
        df.to_csv(os.path.join(self.folder, self.original_hyperpipe_name + '_atlas_mapper_results.csv'))

        # write performance to atlas niftis
        performances = list()

        for roi_name, roi_res in hyperpipe_results.items():
            n_voxels = len(X_extracted[self.roi_indices[roi_name]][0])
            performances.append(np.repeat(roi_res[self.best_config_metric], n_voxels))

        backmapped_img, _, _ = self.neuro_element.inverse_transform(performances)
        backmapped_img.to_filename(os.path.join(self.folder, 'atlas_mapper_performances.nii.gz'))

        if self.create_surface_plots:
            self.surface_plots(backmapped_img)

    def surface_plots(self, perf_img):
        print('Creating surface plots')

        figure, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
        axes = axes.ravel()
        big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

        cnt = 0
        for hemi, infl, sulc, pial in [('left', big_fsaverage.infl_left, big_fsaverage.sulc_left, big_fsaverage.pial_left),
                                        ('right', big_fsaverage.infl_right, big_fsaverage.sulc_right, big_fsaverage.pial_right)]:
            print('Hemi {}'.format(hemi))

            big_texture = surface.vol_to_surf(perf_img, pial, interpolation='nearest')

            for view in ['lateral', 'medial']:
                print('   View {}'.format(view))
                if cnt == 3:
                    output_file = os.path.join(self.folder, 'importance_scores_surface.png')
                else:
                    output_file = None
                plotting.plot_surf_stat_map(infl, big_texture, hemi=hemi, colorbar=True,
                                            title='{} hemisphere {} view'.format(hemi, view),
                                            threshold=0.0001, bg_map=sulc, view=view,
                                            output_file=output_file,
                                            axes=axes[cnt])
                cnt += 1

    def _reshape_roi_data(self, X):
        roi_data = [list() for n in range(len(X[0]))]
        for roi_i in range(len(X[0])):
            for sub_i in range(len(X)):
                roi_data[roi_i].append(X[sub_i][roi_i])
        return roi_data

    def predict(self, X, **kwargs):
        if len(self.hyperpipes_to_fit) == 0:
            raise Exception("No hyperpipes to predict. Did you remember to fit or load the Atlas Mapper?")

        X_extracted, _, _ = self.neuro_element.transform(X)
        X_extracted = self._reshape_roi_data(X_extracted)

        predictions = dict()
        for roi, infos in self.hyperpipe_infos.items():
            roi_index = infos['roi_index']
            predictions[roi] = self.hyperpipes_to_fit[roi].predict(X_extracted[roi_index], **kwargs)
        return predictions

    def load_from_file(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError("Couldn't find atlas mapper meta file")

        self._load(file)

    def load_from_folder(self, folder: str, analysis_name: str = None):
        if not os.path.exists(folder):
            raise NotADirectoryError("{} is not a directory".format(folder))

        if analysis_name:
            meta_file = glob(os.path.join(folder, analysis_name + '_atlas_mapper_meta.json'))
        else:
            meta_file = glob(os.path.join(folder, '*_atlas_mapper_meta.json'))

        if len(meta_file) == 0:
            raise FileNotFoundError("Couldn't find atlas_mapper_meta.json file in {}. Did you specify the correct analysis name?".format(folder))
        elif len(meta_file) > 1:
            raise ValueError("Found multiple atlas_mapper_meta.json files in {}".format(folder))

        self._load(meta_file[0])

    def _load(self, file):
        # load neuro branch
        self.folder = os.path.split(file)[0]
        self.neuro_element = joblib.load(os.path.join(self.folder, 'neuro_element.pkl'))

        with open(file, "r") as read_file:
            self.hyperpipe_infos = json.load(read_file)

        roi_models = dict()
        for roi_name, infos in self.hyperpipe_infos.items():
            model_path = os.path.join(os.path.join(self.folder, infos['hyperpipe_name'] + "_results"), os.path.basename(infos['model_filename']))
            roi_models[roi_name] = Hyperpipe.load_optimum_pipe(model_path)
            self.hyperpipes_to_fit = roi_models
