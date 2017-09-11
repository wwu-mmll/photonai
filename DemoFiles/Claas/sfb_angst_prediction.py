#Abstract
#========
#Fear as an emotional reaction to threatening stimuli or situations is a common feeling with high evolutionary relevance. Fear activates the fight or flight system and should assure survival. Individuals differ in the way they cope with potential threat. Krohne (Krohne, 1989) differentiated in their model of coping between two coping styles: cognitive avoidance and vigilance. Cognitive avoidance is characterized through the avoidance of threat-related information; whereas the vigilant strategy is characterized through an approach an intensive processing of threat-relevant information. Classifying individuals according to their preferred coping style (e.g. identified through the Mainz Coping Inventory, MCI, Krohne & Egloff, 1999), several studies investigated behaviour and neurobiology.  One study showed that in repressors there is a high discrepancy between self-reported and psychophysiological/ behavioral anxiety measures (Derakshan & Eysenck, 2001).#
#
#Unabhängige Daten:
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
#Abhängige Daten:
# - W:\NAE\SPM\2ndLevel_Lisa\ABI_N\Decriptives\ 29170718_Ausgangsdatei_mit Summenscores_Brain
#    - nur die included_HC_all = 1 nehmen
#    - zu prädizierende Variablen:
#       - ABI_Vig (= Vigilanz als Copingstrategie)
#       - Abi_KVermeidung (für kognitive Vermeidung als Strategie)

from Logging.Logger import Logger
import numpy as np
import csv
from os import  listdir

from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch
from PhotonNeuro.BrainAtlas import  BrainAtlas
from PhotonNeuro.AtlasStacker import AtlasStacker, AtlasInfo
from PhotonNeuro.ImageBasics import ResamplingImgs
from sklearn.model_selection import KFold

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
        next(reader) # skip the headline row
        for row in reader:
            abi_vigilanz.append(int(row[2]))
            abi_kognitive_vermeidung.append(int(row[3]))

    return (abi_vigilanz, abi_kognitive_vermeidung)

def loading_mri_ids():
    Logger().info("Loading MRI ids")
    mri_ids = []
    with open('/spm-data/Scratch/spielwiese_claas/sfb-angst/ABI_Daten_Claas.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip the headline row
        for row in reader:
            mri_ids.append(row[0])

    return mri_ids

# Lade abhängige Variablen
#t1_mris = loading_t1_mris()
#t1_amygdala_mris = extract_mri_roi(t1_mris, "amygdala")
#t1_acc_mris = extract_mri_roi(t1_mris, "acc")
#t1_gyrus_cinguli_mris = extract_mri_roi(t1_mris, "gyrus_cinguli")

# Lade unabhängige Variablen
abi_vigilanz, abi_kognitive_vermeidung = loading_abi_variables()
targets = np.asarray(abi_vigilanz)

# Loading nii files
mri_ids = loading_mri_ids()
data = loading_t1_mris(mri_ids)

# preprocess using Photon-Neuro: EVIL
#data = ResamplingImgs(voxel_size=[5, 5, 5], output_img=True).transform(data)

# get single ROI data
#atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Cingulum_Ant_L', 'Cingulum_Ant_R', 'Amygdala_L', 'Amygdala_R'], extraction_mode='vec')
atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Cingulum_Ant_L'], extraction_mode='vec')
myAtlas = BrainAtlas(atlas_info_object=atlas_info)
data = myAtlas.transform(data)


# Building Hyperpipes
# setup photon HP
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error', 'mean_absolute_error', 'r2', 'pearson_correlation', 'variance_explained'],
                    inner_cv=KFold(n_splits=3, shuffle=True, random_state=3),
                    outer_cv=KFold(n_splits=3, shuffle=True, random_state=3),
                    eval_final_performance=True)


# preproc
my_pipe += PipelineElement.create('standard_scaler', {}, test_disabled=True)
my_pipe += PipelineElement.create('pca', {'n_components': [None]}, test_disabled=True)

# estimators
# svr_estimator = PipelineElement.create('SVR', {'kernel': ['linear', 'rbf']})
# tree_estimator = PipelineElement.create('RandomForestRegressor', {'min_samples_split': [2, 5, 10]})
# my_pipe += PipelineSwitch('final_estimator', [svr_estimator, tree_estimator])

my_pipe += PipelineElement.create('RandomForestRegressor', {'min_samples_split': [2, 5, 10]})

# START HYPERPARAMETER SEARCH
my_pipe.fit(data, targets)

rst = my_pipe.result_tree
test = rst.get_best_config_performance_for(outer_cv_fold=0, train_data=False)
best_conf = rst.get_best_config_for(outer_cv_fold=0)
print(test.metrics['variance_explained'])
print(test.metrics['pearson_correlation'])
print()


# # ROI stacking
# atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Cingulum_Ant_L', 'Cingulum_Ant_R', 'Amygdala_L', 'Amygdala_R'], extraction_mode='vec')
# my_pipe += PipelineElement.create('BrainAtlas', {}, atlas_info_object=atlas_info)
# tmp_atlas_stacker = AtlasStacker(atlas_info, [
#     ['pca', {'n_components': [10, 'None']}]
#     ['svc', {'kernel': ['rbf', 'linear']}, {}]
# ])
# my_pipe += PipelineElement('atlas_stacker', tmp_atlas_stacker, {})
# my_pipe += PipelineElement.create('SVR', {'kernel': ['linear', 'rbf']})