# coding=utf-8
# Abstract
# ========
# Fear as an emotional reaction to threatening stimuli or situations is a common feeling with high evolutionary relevance. Fear activates the fight or flight system and should assure survival. Individuals differ in the way they cope with potential threat. Krohne (Krohne, 1989) differentiated in their model of coping between two coping styles: cognitive avoidance and vigilance. Cognitive avoidance is characterized through the avoidance of threat-related information; whereas the vigilant strategy is characterized through an approach an intensive processing of threat-relevant information. Classifying individuals according to their preferred coping style (e.g. identified through the Mainz Coping Inventory, MCI, Krohne & Egloff, 1999), several studies investigated behaviour and neurobiology.  One study showed that in repressors there is a high discrepancy between self-reported and psychophysiological/ behavioral anxiety measures (Derakshan & Eysenck, 2001).#
#
# Unabhängige Daten:
# - T1: W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Data\N_all_Greymatter
# - Funktionelle Kontrastbilder: W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Data\N_all
#    - con_10 für angry versus neutral
#    - con_12 für fearful versus neutral
#    - con17 für Negative (also angry und fearful gemeinsam) versus neutral
# - Zu verwendende ROIs:
#    - Amygdala
#    - Acc
#    - Subgenaules Cingulum, BA25
#
# Abhängige Daten:
# - W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Decriptives\ 29170718_Ausgangsdatei_mit Summenscores_Brain
#    - nur die included_HC_all = 1 nehmen
#    - zu prädizierende Variablen:
#       - ABI_Vig (= Vigilanz als Copingstrategie)
#       - Abi_KVermeidung (für kognitive Vermeidung als Strategie)

import csv
from os import listdir

import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold

from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch
from Logging.Logger import Logger
from PhotonNeuro.AtlasStacker import AtlasInfo
from PhotonNeuro.BrainAtlas import BrainAtlas


def main():
    mri_roi_names = ['Cingulum_Ant_L', 'Cingulum_Ant_R', 'Amygdala_L', 'Amygdala_R']
    for mri_roi_name in mri_roi_names:
        regression_for_mri_roi(mri_roi_name)


        # # ROI stacking
        # atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Cingulum_Ant_L', 'Cingulum_Ant_R', 'Amygdala_L', 'Amygdala_R'], extraction_mode='vec')
        # my_pipe += PipelineElement.create('BrainAtlas', {}, atlas_info_object=atlas_info)
        # tmp_atlas_stacker = AtlasStacker(atlas_info, [
        #     ['pca', {'n_components': [10, 'None']}]
        #     ['svc', {'kernel': ['rbf', 'linear']}, {}]
        # ])
        # my_pipe += PipelineElement('atlas_stacker', tmp_atlas_stacker, {})

        # my_pipe += PipelineElement.create('SVR', {'kernel': ['linear', 'rbf']})


def regression_for_mri_roi(mri_roi_name):
    Logger().info('Perform regression for {}.'.format(mri_roi_name))
    # Lade unabhängige Variablen
    abi_vigilanz, abi_kognitive_vermeidung = loading_abi_variables()
    target_vector = np.asarray(abi_kognitive_vermeidung)

    # Lade abhängige Variablen
    # Loading nii files
    mri_ids = loading_mri_ids()
    source_data = loading_t1_mris(mri_ids)

    # preprocess using Photon-Neuro: EVIL
    # data = ResamplingImgs(voxel_size=[5, 5, 5], output_img=True).transform(data)

    # get single ROI data
    atlas_info = AtlasInfo(atlas_name='AAL', roi_names=[mri_roi_name], extraction_mode='vec')
    roi_atlas = BrainAtlas(atlas_info_object=atlas_info)
    source_data_transformed = roi_atlas.transform(source_data)

    # Building Hyperpipes
    # setup photon HP
    my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                        optimizer_params={},
                        metrics=['mean_squared_error', 'mean_absolute_error', 'r2', 'pearson_correlation',
                                 'variance_explained'], best_config_metric='mean_squared_error',
                        inner_cv=KFold(n_splits=3, shuffle=True, random_state=3),
                        outer_cv=KFold(n_splits=3, shuffle=True, random_state=3),
                        eval_final_performance=True)

    # preproc
    my_pipe += PipelineElement.create('standard_scaler', {}, test_disabled=True)
    my_pipe += PipelineElement.create('pca', {'n_components': [None]}, test_disabled=True)
    my_pipe += PipelineElement.create('SelectPercentile', {}, score_func=f_regression, test_disabled=True)

    # estimators
    svr_estimator = PipelineElement.create('SVR', {'kernel': ['linear', 'rbf'], 'C': [0.7, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]})
    tree_estimator = PipelineElement.create('RandomForestRegressor', {'min_samples_split': [2, 5, 10]})
    #my_pipe += svr_estimator
    my_pipe += PipelineSwitch('final_estimator', [svr_estimator, tree_estimator])

    # START HYPERPARAMETER SEARCH
    my_pipe.fit(source_data_transformed, target_vector)

    result_tree = my_pipe.result_tree
    results_for_test = result_tree.get_best_config_performance_for(outer_cv_fold=0, train_data=False)
    results_for_train = result_tree.get_best_config_performance_for(outer_cv_fold=0, train_data=True)
    best_conf = result_tree.get_best_config_for(outer_cv_fold=0)

    Logger().info("ROI: {0}".format(mri_roi_name))
    Logger().info("Explained variance in test data: {0}; Explained variance in train data: {1}"
                  .format(results_for_test.metrics['variance_explained'],
                          results_for_train.metrics['variance_explained']))
    Logger().info("Correlation for test data: {0}; Correlation for train data: {1}"
                  .format(results_for_test.metrics['pearson_correlation'],
                          results_for_train.metrics['pearson_correlation']))
    _write_to_file("-------------------------------------------------------------")
    _write_to_file("ROI: {0}".format(mri_roi_name))
    _write_to_file("Configuration: {}".format(best_conf.config_dict))
    _write_to_file("Explained variance in test data: {0}"
                   .format(results_for_test.metrics['variance_explained']))
    _write_to_file("Correlation for test data: {0}"
                   .format(results_for_test.metrics['pearson_correlation']))
    _write_to_file("Explained variance in train data: {0}"
                   .format(results_for_train.metrics['variance_explained']))
    _write_to_file("Correlation for train data: {0}"
                   .format(results_for_train.metrics['pearson_correlation']))


def loading_abi_variables():
    # Todo
    #  - W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Decriptives\ 29170718_Ausgangsdatei_mit Summenscores_Brain
    #    - nur die included_HC_all = 1 nehmen
    #    - zu prädizierende Variablen:
    #      - ABI_Vig (= Vigilanz als Copingstrategie)
    #      - Abi_KVermeidung (für kognitive Vermeidung als Strategie)

    Logger().info("Loading ABI variables")
    abi_vigilanz = []
    abi_kognitive_vermeidung = []
    with open('/spm-data/Scratch/spielwiese_claas/sfb-angst/ABI_Daten_Claas.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the headline row
        for row in reader:
            abi_vigilanz.append(int(row[2]))
            abi_kognitive_vermeidung.append(int(row[3]))

    return (abi_vigilanz[:200], abi_kognitive_vermeidung[:200])


def loading_mri_ids():
    Logger().info("Loading MRI ids")
    mri_ids = []
    with open('/spm-data/Scratch/spielwiese_claas/sfb-angst/ABI_Daten_Claas.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the headline row
        for row in reader:
            mri_ids.append(row[0])

    return mri_ids[:200]


def loading_t1_mris(mri_ids):
    # /spm-data/vault-data2/NAE/SPM/2ndLevel_Lisa/ABI_N/Data/N_all_Greymatter/
    Logger().info("Loading T1 MRIs")
    root_folder = '/spm-data/vault-data2/NAE/SPM/2ndLevel_Lisa/ABI_N/Data/N_all_Greymatter/'
    list_of_files = []
    # todo: extrem ugly
    for mri_id in mri_ids:
        for filename in listdir(root_folder):
            if mri_id in filename and filename.endswith(".nii") and filename.startswith("s"):
                list_of_files.append(root_folder + filename)
    return list_of_files


def _write_to_file(line: str):
    with open("sfb_angst_result.log", "a", newline='\n') as text_file:
        text_file.write('\n')
        text_file.write(line)


main()
