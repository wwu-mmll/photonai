import glob

import pandas
from sklearn.model_selection import KFold

from Framework import LogExtractor
from Framework.PhotonBase import PipelineElement, Hyperpipe
from Logging.Logger import Logger
from PhotonNeuro.AtlasStacker import AtlasInfo
from PhotonNeuro.BrainAtlas import BrainAtlas
from pathlib import Path
import numpy as np

ROOT_DIR = "/spm-data/Scratch/spielwiese_claas"
def load_etc_subject_ids_and_targets(xls_file_path: str):
    # subject IDs
    subject_ids = pandas.read_excel(open(target_xls_file, 'rb'), sheetname='ECT', parse_cols="A", squeeze=True)
    # load ECT targets
    targets = pandas.read_excel(open(xls_file_path, 'rb'), sheetname='ECT', parse_cols="I", squeeze=True)
    Logger().debug(targets)
    return (subject_ids, targets)

def load_dti_images(folder_path: str, subject_ids):
    # load features (imgs)
    tmp_files = list()
    for subject_id in subject_ids:
        tmp = glob.glob(folder_path + '/' + subject_id + '*.gz')
        tmp_files.append(tmp)
    dti_image_files = [item for sublist in tmp_files for item in sublist]
    Logger().debug(dataset_files)
    return dti_image_files

def extract_brain_from_dti_images(dti_image_files):
    atlas_info = AtlasInfo(atlas_name='mni_icbm152_t1_tal_nlin_sym_09a_mask', mask_threshold=.5,
                       roi_names='all', extraction_mode='vec')
    brain_atlas = BrainAtlas(atlas_info_object=atlas_info)
    return brain_atlas.transform(X=dti_image_files)

def extract_unique_dti_features(unstripped_data: np.array):
    Logger().info("Extracting unique features.")
    columns_to_delete = list()
    for j in range(0, unstripped_data.shape[1]):
        ref_value = unstripped_data[0, j]
        equal = True
        for i in range(0, len(unstripped_data[:,j])):
            if unstripped_data[i, j] != ref_value:
                equal = False
                break
        if equal:
           columns_to_delete.append(j)
    Logger().info("Element wise equal columns: {0}, {1} will be left.".format(len(columns_to_delete), unstripped_data.shape[1] - len(columns_to_delete)))
    return np.delete(unstripped_data, np.asarray(columns_to_delete), 1)

subject_ids, targets = load_etc_subject_ids_and_targets(ROOT_DIR + '/Key_Information_ECT_sample_20170829.xlsx')

cached_preprocessed_dti_data_file = Path(ROOT_DIR + "/cached_preprocessed_dti_data.npy")
if cached_preprocessed_dti_data_file.exists():
    dti_roi_brain_striped = np.load(cached_preprocessed_dti_data_file)
else:
    dti_image_files = load_dti_images(subject_ids, ROOT_DIR + '/ECT_data_for_Tim/MD/')
    dti_roi_brain = extract_brain_from_dti_images(dti_image_files)
    dti_roi_brain_striped = extract_unique_dti_features(dti_roi_brain)
    np.save(cached_preprocessed_dti_data_file, dti_roi_brain_striped)



# setup photon HP
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error', 'mean_absolute_error'],
                    inner_cv=KFold(n_splits=2, shuffle=True, random_state=3),
                    outer_cv=KFold(n_splits=2, shuffle=True, random_state=2),
                    eval_final_performance=True)

#my_pipe += PipelineElement.create('SmoothImgs', {'fwhr': [[8, 8, 8], [12, 12, 12]]})
#my_pipe += PipelineElement.create('ResampleImgs', {'voxel_size': [[5, 5, 5]]})


#atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='box')
#my_pipe += PipelineElement.create('BrainAtlas', {}, atlas_info_object=atlas_info)
# my_pipe += PipelineElement('atlas_stacker',
#                            AtlasStacker(atlas_info, [['SVR', {'kernel': ['rbf', 'linear']}, {}]]),
#                            {})

my_pipe += PipelineElement.create('SVR', {'kernel': ['linear']})

# START HYPERPARAMETER SEARCH
my_pipe.fit(dti_roi_brain_striped, targets)


# resImg = ResamplingImgs(voxel_size=[5, 5, 5], output_img=True).transform(dataset_files)
# smImg = SmoothImgs(fwhr=[6, 6, 6], output_img=True).transform(resImg)
#
# from Photon_Neuro0.BrainAtlas import BrainAtlas
# myAtlas = BrainAtlas(atlas_name='AAL',

#                      extract_mode='vec',
#                      whichROIs='all')
# roi_data = myAtlas.transform(X=smImg)
# myAtlas.getInfo()

logex = LogExtractor(my_pipe.results)
logex.etract_csv('jhonny_dti_results.csv')

# # remove all cols with only 0s
# is_in_mat = []
# for j in range(0, single_roi.shape[1]):
#     is_in = not np.any(single_roi[:, j])
#     is_in_mat.append(is_in)
# single_roi = single_roi[:, is_in_mat]