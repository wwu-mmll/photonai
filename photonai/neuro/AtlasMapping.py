from photonai.neuro.AtlasStacker import AtlasInfo
from photonai.neuro.BrainAtlas import BrainAtlas, AtlasLibrary
from photonai.base.PhotonBase import Hyperpipe
from photonai.photonlogger.Logger import Logger
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
import pandas as pd
import multiprocessing as mp
import numpy as np
import os


class AtlasMapper:

    def generate_mappings(self, hyperpipe, folder):

        roi_list = list()
        target_element_name = ""

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
            pass

        # then check usual pipeline_elements for a) NeuroModuleBranch -> check children and b) BrainAtlas directly
        for element in hyperpipe.pipeline_elements:
            if isinstance(element.base_element, BrainAtlas):
                target_element_name = element.name
                roi_list = found_brain_atlas(element)

        hyperpipes_to_fit = list()

        if len(roi_list) > 0:
            for param in roi_list:
                copy_of_hyperpipe = hyperpipe.copy_me()
                copy_of_hyperpipe.set_params({target_element_name + "__rois": param})
                # mkdir could be needed?
                copy_of_hyperpipe.output_settings.project_folder = os.path.join(folder, "_" + param)
                hyperpipes_to_fit.append(copy_of_hyperpipe)
        else:
            raise Exception("No Rois found...")

    def fit(self, X, y=None, **kwargs):
        pass

    def predict(self, X, **kwargs):
        pass

class AtlasMapping:

    def __init__(self, atlas_info, hyperpipe_constructor, write_to_folder, n_processes=1, write_summary_to_excel=True, savePipes=''):
        self.savePipes = savePipes
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
        #pipe._set_verbosity(-1)
        return pipe

    def fit(self, dataset_files, targets, TIV=None):
        """
        This function takes MRI (e.g. nifti) images and targets and optimizes the same hyperpipe in each region of an atlas independently
        :param dataset_files: list of absolute paths to MRI files (e.g. nifti or analyze)
        :param targets: targets for supervised learning
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

                # apply individual TIV correction (cf. Gaser mail and VBM8)
                if TIV is not None:
                    print('Removing TIV...')
                    roi_data_new = np.empty(roi_data.shape)
                    for i in range(roi_data.shape[0]):
                        gaserFactor = 1520 / TIV[i]  # 1530 is meanTIV from IXI dataset
                        roi_data_new[i, :] = roi_data[i, :] * gaserFactor
                    roi_data = roi_data_new

                # get pipeline and fit
                my_hyperpipe = AtlasMapping.get_pipe(pipe=self.hyperpipe_constructor(),
                                                     write_to_folder=self.write_to_folder,
                                                     new_pipe_name=roi_label + '_pipe')
                my_hyperpipe.fit(data=roi_data, targets=targets)

                if self.savePipes != '':
                    my_hyperpipe.save_optimum_pipe(file=self.write_to_folder + self.savePipes + roi_label + '_model.photon')

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

    @staticmethod
    def predict(dataset_files, brainAtlas, model_folder):
        """
        This function takes an atlas info object and returns a prediction per region and sample if a PHOTON model for the ROI can be found
        :param atlas_info: The PHOTON Neuro atlas_info object containing details of the atlas and regions to process
        :param model_folder: absolute path of the folder containing the PHOTON models trained using AtlasMapping.fit()
        :return: prediction per region and sample
        """
        predictions = dict()
        for roi_label in atlas_info.roi_names:
            # load the PHOTON model
            my_model = Hyperpipe.load_optimum_pipe(model_folder + roi_label + '_model.photon')
            # my_model.get_params()  # get model details

            # for each Region of Interest
            Logger().info(roi_label)
            # get ROI info
            atlas = BrainAtlas(atlas_info_object=brainAtlas)
            roi_atlas_info = AtlasInfo(atlas_name=atlas.atlas_name, roi_names=[roi_label],
                                       extraction_mode=atlas.extract_mode)
            roi_atlas_object = BrainAtlas(atlas_info_object=roi_atlas_info)
            roi_data = np.squeeze(roi_atlas_object.transform(dataset_files))

            # # apply individual TIV correction (cf. Gaser mail and VBM8)
            # if TIV is not None:
            #     print('Removing TIV...')
            #     roi_data_new = np.empty(roi_data.shape)
            #     for i in range(roi_data.shape[0]):
            #         gaserFactor = 1520 / TIV[i]  # 1530 is meanTIV from IXI dataset
            #         roi_data_new[i, :] = roi_data[i, :] * gaserFactor
            #     roi_data = roi_data_new

            # predict label for new data using this model
            predictions[roi_label] = my_model.predict(roi_data)

        return predictions


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

if __name__ == "__main__":
    # get data
    where = 'remote'
    if where == 'home':
        folder = 'D:/myGoogleDrive/work/all_skripts/py_code/checks/ageResults/'
    if where == 'work':
        folder = 'C:/Users/hahnt/myGoogleDrive/work/all_skripts/py_code/checks/ageResults/'
    if where == 'remote':
        folder = '/spm-data/Scratch/spielwiese_tim/checks/ageResults/'

    dataset_files = np.load(file=folder+'img_data_resampled_' + str(3) + '.npy')


    # img_file = [
    #     '/spm-data/vault-data1/ImageDatabase_2018/CAT12 r1184/FOR2107/mri/mwp10001_t1mprsagp2iso20140911PSYCHMACS0001A1s003a1001.nii']  # df index 0
    # age = 26

    # img_file = '/spm-data/vault-data1/ImageDatabase_2018/CAT12 r1184/MuensterNeuroimagingCohort/mri/mwp1N0969_NAE_N169.nii' # df index 1000
    # age = 23

    # img_file = '/spm-data/vault-data1/ImageDatabase_2018/CAT12 r1184/BiDirect/mri/mwp1BD30354_T1_BD30354.nii'   # df index 1500
    # age = 49.6

    # get info for the atlas
    # BrainAtlas.whichAtlases()

    # atlas_info = AtlasInfo(atlas_name='HarvardOxford-cort-maxprob-thr50', roi_names='all', extraction_mode='mean')
    # atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='vec')
    atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Precentral_L', 'Frontal_Sup_R'], extraction_mode='vec')
    # atlas_info = AtlasInfo(atlas_name='AAL', roi_names=[2001, 2002, 2102], extraction_mode='vec')
    # atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='box')
    #atlas = 'HarvardOxford-cort-maxprob-thr25'
    #atlas = 'HarvardOxford-sub-maxprob-thr25'
    #atlas = 'AAL'
    #atlas_info = AtlasInfo(atlas_name=atlas, roi_names='all', extraction_mode='vec')

    model_folder = folder + '/AAL/'
    test = AtlasMapping.predict(dataset_files=dataset_files, brainAtlas=atlas_info, model_folder=model_folder)

    debug = True
