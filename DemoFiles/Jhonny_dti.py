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
from sklearn.model_selection import LeaveOneOut
import pandas as pd

ROOT_DIR = "/spm-data/Scratch/spielwiese_claas"

def load_etc_subject_ids_and_targets(xls_file_path: str):
    # subject IDs
    subject_ids = pandas.read_excel(open(xls_file_path, 'rb'), sheetname='ECT', parse_cols="A", squeeze=True)
    # load ECT targets
    targets = pandas.read_excel(open(xls_file_path, 'rb'), sheetname='ECT', parse_cols="I", squeeze=True)
    targets = targets.as_matrix()
    Logger().debug(targets)
    return (subject_ids, targets)

def load_dti_images(folder_path: str, subject_ids):
    # load features (imgs)
    tmp_files = list()
    for subject_id in subject_ids:
        tmp = glob.glob(folder_path + '/' + subject_id + '*.gz')
        tmp_files.append(tmp)
    dti_image_files = [item for sublist in tmp_files for item in sublist]
    Logger().debug(str(dti_image_files))
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

def load_and_preprocess_dti_data():
    cached_preprocessed_dti_data_file = Path(ROOT_DIR + "/cached_preprocessed_dti_data.npy")
    if cached_preprocessed_dti_data_file.exists():
        Logger().info("Loading cached preprocessed DTI data...")
        dti_roi_brain_striped = np.load(cached_preprocessed_dti_data_file)
    else:
        dti_image_files = load_dti_images(ROOT_DIR + '/ECT_data_for_Tim/MD/', subject_ids)
        dti_roi_brain = extract_brain_from_dti_images(dti_image_files)
        dti_roi_brain_striped = extract_unique_dti_features(dti_roi_brain)
        dti_roi_brain_striped = np.nan_to_num(dti_roi_brain_striped)
        np.save(cached_preprocessed_dti_data_file, dti_roi_brain_striped)
    return dti_roi_brain_striped

def classical_classification() -> Hyperpipe:
    # Strategie 1 - klassisch:
    # - 2 x CV leave 1 out. (n<100)
    # - Normieren ([0, ..., 1], keine, standard scaler)
    # - PCA (lossless, aus)
    # - Feature selection mit
    #    - Lasso (an, aus)
    #    - F-Classification
    # - Classifier:
    #    - RF
    #    - SVM
    #    - GPC
    #    - naive Bayes
    pipe = Hyperpipe('primary_pipe', optimizer='timeboxed_random_grid_search',
                optimizer_params={"limit_in_minutes": 2},
                metrics=['mean_squared_error'],
                inner_cv=LeaveOneOut(),
                outer_cv=LeaveOneOut(),
                eval_final_performance=True)
    # pipe = Hyperpipe('primary_pipe', optimizer='timeboxed_random_grid_search',
    #             optimizer_params={"limit_in_minutes": 0.05},
    #             metrics=['mean_squared_error', 'pearson_correlation'],
    #             inner_cv=KFold(n_splits=3, shuffle=True),
    #             outer_cv=KFold(n_splits=3, shuffle=True),
    #             eval_final_performance=True)
    standard_scaler = PipelineElement.create("standard_scaler", hyperparameters={}, test_disabled=True)
    #min_max_scaler = PipelineElement.create("standard_scaler", hyperparameters={}, test_disabled=True)
    pipe += standard_scaler
    pipe += PipelineElement.create("pca", hyperparameters={}, test_disabled=True, n_components=None)
    pipe += PipelineElement.create("f_regression_select_percentile", hyperparameters={"percentile": [10, 50]}, test_disabled=True)

    #svr = PipelineElement.create("SVR", hyperparameters={"kernel":["rbf", "linear"], "C":[0.5, 1, 2]})
    rndf = PipelineElement.create("RandomForestRegressor", hyperparameters={"min_samples_leaf": [1, 5], })
    pipe += rndf

    return pipe

subject_ids, targets = load_etc_subject_ids_and_targets(ROOT_DIR + '/Key_Information_ECT_sample_20170829.xlsx')
dti_preprocessed = load_and_preprocess_dti_data()
pipe = classical_classification()

pipe.fit(np.ascontiguousarray(dti_preprocessed),targets)

logex = LogExtractor.LogExtractor(pipe.result_tree)
logex.extract_csv('johnny_dti_results.csv')

