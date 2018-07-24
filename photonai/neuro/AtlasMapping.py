from photonai.neuro.AtlasStacker import AtlasInfo
from photonai.neuro.BrainAtlas import BrainAtlas
from photonai.photonlogger.Logger import Logger
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
import pandas as pd


class AtlasMapping():
    def _get_pipe(hyperpipe, write_to_folder, pipe_name):
        """
        This function takes a hyperpipe, adjusts the name and results file name/path and returns the new hyperpipe
        :param hyperpipe: hyperpipe provided by the user
        :param write_to_folder: absolute path and filename of the results folder
        :param pipe_name: name of the hyperpipe (adjusted according to ROI)
        :return: the updated hyperpipe
        ToDo: Deepcopy hyperpipe to be sure the ROI pipes are independent
        """
        hyperpipe.name = pipe_name
        hyperpipe.persist_options.local_file = write_to_folder + 'results_' + pipe_name + '.p'
        return hyperpipe

    @staticmethod
    def mapAtlas(dataset_files, targets, hyperpipe, atlas_info, write_to_folder, write_summary_to_excel=True):
        """
        This function takes MRI (e.g. nifti) images and targets and optimizes the same hyperpipe in each region of an atlas independently
        :param dataset_files: list of absolute paths to MRI files (e.g. nifti or analyze)
        :param targets: targets for supervised learning
        :param atlas_info: The PHOTON Neuro atlas_info object cnntaining details of the atlas and regions to process
        :param write_to_folder: output folder for all results
        :param write_summary_to_excel: write results to an MS Excel file
        :return: results summary across Regions of Interest as a pandas dataframe
        ToDo: get labels_applied more elegantly
        """
        # get all relevant Regions of Interest
        atlas_object = BrainAtlas(atlas_info_object=atlas_info)
        # atlas_object.getInfo()
        atlas_object.transform(dataset_files[0:1])  # to get labels_applied

        # for each Region of Interest
        res_list = list()
        for roi_label in atlas_object.labels_applied:
            Logger().info(roi_label)

            # get ROI info
            roi_atlas_info = AtlasInfo(atlas_name=atlas_object.atlas_name, roi_names=[roi_label],
                                       extraction_mode=atlas_object.extract_mode)
            roi_atlas_object = BrainAtlas(atlas_info_object=roi_atlas_info)
            roi_data = roi_atlas_object.transform(dataset_files)

            # get pipeline and fit
            my_hyperpipe = AtlasMapping._get_pipe(write_to_folder=write_to_folder, hyperpipe=hyperpipe,
                                                  pipe_name=roi_label + '_pipe')
            my_hyperpipe.fit(data=roi_data, targets=targets)

            # get summary of results
            res_file = my_hyperpipe.mongodb_writer.save_settings.local_file
            res_tmp = ResultsTreeHandler(res_file).get_performance_table().tail(n=1).drop(['fold', 'n_train'], axis=1)
            res_tmp.insert(loc=0, column='Region', value=roi_label)
            res_list.append(res_tmp)

        results_summary = pd.concat(res_list)

        if write_summary_to_excel:
            results_summary.to_excel(write_to_folder + 'results_summary_brainMap.xlsx', index=False)

        return results_summary
