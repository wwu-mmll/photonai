import glob

import pandas
from sklearn.model_selection import KFold

from Framework import LogExtractor
from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch
from Logging.Logger import Logger
from PhotonNeuro.AtlasStacker import AtlasInfo
from PhotonNeuro.BrainAtlas import BrainAtlas
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import pickle
from sklearn.model_selection import LeaveOneOut, ShuffleSplit
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('Agg')

ROOT_DIR = "/spm-data/Scratch/spielwiese_claas"

def load_etc_subject_ids_and_targets(xls_file_path: str):
    # subject IDs
    subject_ids = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="A", squeeze=True)
    # load ECT targets
    targets = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="I", squeeze=True)
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
    pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                     metrics=['mean_squared_error'],
                     best_config_metric='mean_squared_error',
                     inner_cv=LeaveOneOut(),
                     eval_final_performance=False)

    standard_scaler = PipelineElement.create("standard_scaler", hyperparameters={}, test_disabled=True)

    pipe += standard_scaler
    pipe += PipelineElement.create("pca", hyperparameters={}, test_disabled=True, n_components=None)
    pipe += PipelineElement.create("f_regression_select_percentile", hyperparameters={"percentile": [10, 30, 50, 70]}, test_disabled=True)

    svr = PipelineElement.create("SVR", hyperparameters={"kernel": ["rbf", "linear"], "C": [0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 5]})
    # rndf = PipelineElement.create("RandomForestRegressor", hyperparameters={"min_samples_leaf": [1, 5]}, test_disabled=False)
    # pipe += PipelineSwitch('final_estimator', [svr, rndf])
    # pipe += rndf
    pipe += svr

    return pipe

subject_ids, targets = load_etc_subject_ids_and_targets(ROOT_DIR + '/Key_Information_ECT_sample_20170829.xlsx')
dti_preprocessed = load_and_preprocess_dti_data()
pipe = classical_classification()

corr_coef = []
corr_p = []

for i in range(dti_preprocessed.shape[1]):
    feature = dti_preprocessed[:, i]
    # corr = np.correlate(feature, targets)
    corr = pearsonr(feature, targets)
    corr_coef.append(corr[0])
    corr_p.append(corr[1])

corr_coef = np.array(corr_coef)
corr_p = np.array(corr_p)
idx_of_significant = np.where(corr_p <= 0.01)
# nr_of_significants = idx_of_significant[0].shape[0]

# histo_vals = np.histogram(corr_coef[idx_of_significant])
# plt.hist(corr_coef[idx_of_significant])
# plt.show()

# tmp_RFR = PipelineElement.create("RandomForestRegressor", hyperparameters={"min_samples_leaf": [1, 5], },
#                                   test_disabled=False)
# tmp_RFR.fit(dti_preprocessed, targets)
# # forest_feature_importance
# most_important_features_index = np.argsort(tmp_RFR.base_element.feature_importances_)
# most_important_features = dti_preprocessed[:, most_important_features_index[:200]]
# num_features = dti_preprocessed.shape[1]
# fig = plt.figure()
# plt.plot(np.linspace(0, num_features, num_features), tmp_RFR.base_element.feature_importances_)
# plt.show()
#
# tmp_PCA = PipelineElement.create('pca', {}, test_disabled=False, n_components=2)
# tmp_PCA.fit(most_important_features)
# explained_variance = np.sum(tmp_PCA.base_element.explained_variance_)
# print(explained_variance)
# dti_preprocessed_three_dim = tmp_PCA.transform(most_important_features)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dti_preprocessed_three_dim[:, 0], dti_preprocessed_three_dim[:, 1], targets)
# plt.show()


features = np.ascontiguousarray(np.array(np.squeeze(dti_preprocessed[:, idx_of_significant])))
pipe.fit(features, targets)

# new_predictions = pipe.predict(features)
# pred_corr = pearsonr(new_predictions, targets)
# print(pred_corr)

config_pearsons_list = []

search_tree = pipe.result_tree.config_list[0].fold_list[0].train.config_list
for config_item in search_tree:
    pearson_list_y_true = []
    pearson_list_y_pred = []
    for fold in config_item.fold_list:
        pearson_list_y_true.append(fold.test.y_true)
        pearson_list_y_pred.append(fold.test.y_predicted)
    pred_pearson = pearsonr(pearson_list_y_pred, pearson_list_y_true)
    print(pred_pearson)

logex = LogExtractor.LogExtractor(pipe.result_tree)
logex.extract_csv('johnny_dti_results.csv')

# pickle.dump(pipe, open('jonny_pipe.p', 'wb'))




