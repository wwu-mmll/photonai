from photonai.neuro.AtlasStacker import AtlasInfo
from photonai.neuro.BrainAtlas import BrainAtlas
from photonai.photonlogger.Logger import Logger
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
import pandas as pd
import multiprocessing as mp


class AtlasMapping:

    def __init__(self, atlas_info, hyperpipe_constructor, write_to_folder, n_processes=1, write_summary_to_excel=True):
        self.atlas_info = atlas_info
        self.hyperpipe_constructor = hyperpipe_constructor
        self.pipe = self.hyperpipe_constructor()
        self.write_to_folder = write_to_folder
        self.n_processes = n_processes
        self.write_summary_to_excel = write_summary_to_excel
        self.res_list = list()

    @staticmethod
    def get_pipe(pipe, new_pipe_name, write_to_folder):
        """
        This function takes a hyperpipe, adjusts the name and results file name/path and returns the new hyperpipe
        :param pipe: hyperpipe provided by the user
        :param write_to_folder: absolute path and filename of the results folder
        :param pipe_name: name of the hyperpipe (adjusted according to ROI)
        :return: the updated hyperpipe
        """
        pipe.name = new_pipe_name
        pipe.persist_options.local_file = write_to_folder + 'results_' + new_pipe_name + '.p'
        pipe._set_verbosity(-1)
        return pipe

    def fit(self, dataset_files, targets):
        """
        This function takes MRI (e.g. nifti) images and targets and optimizes the same hyperpipe in each region of an atlas independently
        :param dataset_files: list of absolute paths to MRI files (e.g. nifti or analyze)
        :param targets: targets for supervised learning
        :param atlas_info: The PHOTON Neuro atlas_info object containing details of the atlas and regions to process
        :param write_to_folder: output folder for all results
        :param write_summary_to_excel: write results to an MS Excel file
        :return: results summary across Regions of Interest as a pandas dataframe
        ToDo: get labels_applied more elegantly
        """
        if self.n_processes > 1:
            results_summary = self.mapAtlas_para(dataset_files, targets)
        else:
            # get all relevant Regions of Interest
            atlas_object = BrainAtlas(atlas_info_object=self.atlas_info)
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
                my_hyperpipe = AtlasMapping.get_pipe(pipe=self.hyperpipe_constructor(),
                                                     write_to_folder=self.write_to_folder,
                                                     new_pipe_name=roi_label + '_pipe')
                my_hyperpipe.fit(data=roi_data, targets=targets)

                # get summary of results
                res_file = my_hyperpipe.mongodb_writer.save_settings.local_file
                res_tmp = ResultsTreeHandler(res_file).get_performance_table().tail(n=1).drop(['fold', 'n_train'], axis=1)
                res_tmp.insert(loc=0, column='Region', value=roi_label)
                res_list.append(res_tmp)

            results_summary = pd.concat(res_list)

            if self.write_summary_to_excel:
                results_summary.to_excel(self.write_to_folder + 'results_summary_brainMap.xlsx', index=False)

        return results_summary

    def mapAtlas_para(self, dataset_files, targets):
        # get all relevant Regions of Interest
        atlas_object = BrainAtlas(atlas_info_object=self.atlas_info)
        # atlas_object.getInfo()
        atlas_object.transform(dataset_files[0:1])  # to get labels_applied

        # Run parallel pool over all regions of interest
        self.run_parallelized_hyperpipes(atlas_object=atlas_object, dataset_files=dataset_files, targets=targets)
        roi_performance = pd.concat(self.res_list)
        return roi_performance

    def run_parallelized_hyperpipes(self, atlas_object, dataset_files, targets):
        pool = mp.Pool(processes=self.n_processes)
        for roi_label in atlas_object.labels_applied:
            pool.apply_async(run_parallelized_mapping,
                             args=(self.hyperpipe_constructor, dataset_files, targets, roi_label,
                                   atlas_object, self.write_to_folder),
                             callback=self.collect_results)
        pool.close()
        pool.join()

    def collect_results(self, result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        self.res_list.append(result)


def run_parallelized_mapping(hyperpipe_constructor, dataset_files, targets, roi_label, atlas_object, write_to_folder):
    # Create new instance of hyperpipe and set all parameters
    Logger().info(roi_label)

    # get ROI info
    roi_atlas_info = AtlasInfo(atlas_name=atlas_object.atlas_name, roi_names=[roi_label],
                               extraction_mode=atlas_object.extract_mode)
    roi_atlas_object = BrainAtlas(atlas_info_object=roi_atlas_info)
    roi_data = roi_atlas_object.transform(dataset_files)

    # get pipeline and fit
    my_hyperpipe = AtlasMapping.get_pipe(pipe=hyperpipe_constructor(), write_to_folder=write_to_folder,
                                         new_pipe_name=roi_label + '_pipe')
    my_hyperpipe.fit(data=roi_data, targets=targets)

    # get summary of results
    res_file = my_hyperpipe.mongodb_writer.save_settings.local_file
    res_tmp = ResultsTreeHandler(res_file).get_performance_table().tail(n=1).drop(['fold', 'n_train'], axis=1)
    res_tmp.insert(loc=0, column='Region', value=roi_label)

    return res_tmp
