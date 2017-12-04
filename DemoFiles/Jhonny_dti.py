import matplotlib
# import matplotlib.rcsetup as rcsetup
# print(rcsetup.all_backends)
# matplotlib.use('Qt5Agg')

import glob
import pickle
import pandas
from sklearn.model_selection import KFold

from Framework import LogExtractor
from Framework.PhotonBase import PipelineElement, Hyperpipe, PipelineSwitch
from Logging.Logger import Logger
from PhotonNeuro.AtlasStacker import AtlasInfo
from PhotonNeuro.BrainAtlas import BrainAtlas
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from sklearn.model_selection import LeaveOneOut, ShuffleSplit

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    pipe += PipelineElement.create("pearson_feature_selector", hyperparameters={'p_threshold': [0.05, 0.01, 0.001]}, test_disabled=False)
    pipe += PipelineElement.create("standard_scaler", hyperparameters={}, test_disabled=False)
    pipe += PipelineElement.create("pca", hyperparameters={}, test_disabled=True, n_components=None)
    svr = PipelineElement.create("SVR", hyperparameters={'kernel': ['linear', 'rbf'], 'C': [0.5, 1, 2]})
    pipe += svr

    # svr = PipelineElement.create("SVR", hyperparameters={"kernel": ["rbf", "linear"], "C": [0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 5]})
    # rndf = PipelineElement.create("RandomForestRegressor", hyperparameters={"min_samples_leaf": [1, 5]}, test_disabled=False)
    # pipe += PipelineSwitch('final_estimator', [svr, rndf])
    # pipe += rndf

    return pipe


def fit_model(targets):
    # Load Data and create pipeline
    dti_preprocessed = load_and_preprocess_dti_data()
    pipe = classical_classification()

    features = np.ascontiguousarray(np.array(np.squeeze(dti_preprocessed)))
    Logger().info('Starting Hyperparameter Search')
    pipe.fit(features, targets)

    Logger().info('Calculating output correlations & stuff')
    config_pearsons_list = []
    mse_list = []
    indices_list = []

    search_tree = pipe.result_tree.config_list[0].fold_list[0].train.config_list
    for config_item in search_tree:
        pearson_list_y_true = []
        pearson_list_y_pred = []
        for fold in config_item.fold_list:
            pearson_list_y_true.append(fold.test.y_true[0])
            pearson_list_y_pred.append(fold.test.y_predicted[0])
            mse = fold.test.metrics['mean_squared_error']
            mse_list.append(mse)
            indices_list.append(fold.test.indices[0])
        pred_pearson = pearsonr(pearson_list_y_pred, pearson_list_y_true)
        print(pred_pearson)
        print(np.mean(mse_list))
        print(np.std(mse_list))

        pickle.dump(pearson_list_y_true, open('jonny_pipe_y_true.p', 'wb'))
        pickle.dump(pearson_list_y_pred, open('jonny_pipe_y_pred.p', 'wb'))

    logex = LogExtractor.LogExtractor(pipe.result_tree)
    logex.extract_csv('johnny_dti_results.csv')



def evaluate_predictions(targets):
    pearson_list_y_true = pickle.load(open('jonny_pipe_y_true.p', 'rb'))#
    pearson_list_y_pred = pickle.load(open('jonny_pipe_y_pred.p', 'rb'))

    responder = pandas.read_excel(open(xls_file, 'rb'), sheet_name='ECT', usecols="K", squeeze=True)
    hamilton_pre = pandas.read_excel(open(xls_file, 'rb'), sheet_name='ECT', usecols="G", squeeze=True)
    hamilton_post = pandas.read_excel(open(xls_file, 'rb'), sheet_name='ECT', usecols="H", squeeze=True)

    hamilton_post_predicted = np.subtract(hamilton_pre, pearson_list_y_pred)
    index_post_predicted = np.where(hamilton_post_predicted <= 8)
    index_responders = np.where(responder == 1)
    predicted_responders = np.zeros(targets.size)
    predicted_responders[index_post_predicted] = 1

    print(accuracy_score(y_true=responder, y_pred=predicted_responders))
    print(classification_report(y_true=responder, y_pred=predicted_responders))

    pickle.dump((hamilton_post, hamilton_post_predicted, responder, predicted_responders), open('jonny_pipe_others.p', 'wb'))


    # print(hamilton_post_predicted)
    # print(index_post_predicted)
    # print(index_responders)



#
#
# xls_file = ROOT_DIR + '/Key_Information_ECT_sample_20170829.xlsx'
# subject_ids, targets = load_etc_subject_ids_and_targets(xls_file)

# fit_model(targets)
# evaluate_predictions(targets)
#
loaded_data = pickle.load(open('jonny_pipe_others.p', 'rb'))
pearson_list_y_true = pickle.load(open('jonny_pipe_y_true.p', 'rb'))
pearson_list_y_pred = pickle.load(open('jonny_pipe_y_pred.p', 'rb'))
#
# sort_index = np.argsort(pearson_list_y_true)
#
# pearson_list_y_true = np.array(pearson_list_y_true)[sort_index]
# pearson_list_y_pred = np.array(pearson_list_y_pred)[sort_index]

hamilton_post = loaded_data[0]
hamilton_post_predicted = loaded_data[1]


rang = spearmanr(hamilton_post, hamilton_post_predicted)
print(rang)

predicted_responders = loaded_data[3]
responder = loaded_data[2]

always_fifteen = np.ones(hamilton_post.shape) * 12.5
mse_always_fifteen = mean_squared_error(y_true=pearson_list_y_true, y_pred=always_fifteen)
print(mse_always_fifteen)

print(mean_squared_error(y_true=pearson_list_y_true, y_pred=pearson_list_y_pred))
# hamilton_post_predicted = hamilton_post_predicted[sort_index]
# hamilton_post = hamilton_post[sort_index]
# responder = responder[sort_index]
# predicted_responders = predicted_responders[sort_index]
#
# correct_predictions = np.where(responder == predicted_responders)
# success = np.zeros(predicted_responders.shape)
# success[correct_predictions] = 1
#
# fig, axes = plt.subplots(3, 1, sharex=True)
# x = np.arange(0, hamilton_post.size, 1)
# axes[1].plot(x, hamilton_post, c='r')
# axes[1].plot(x, hamilton_post_predicted, c='b')
# axes[1].set_title('Hamilton Post EKT')
# axes[0].plot(x, pearson_list_y_true, c='r')
# axes[0].plot(x, pearson_list_y_pred, c='b')
# axes[0].set_title('Hamilton Delta')
# axes[2].scatter(x, success, c='black')
# axes[2].set_title('Successfully classified Hamilton < 8 Responders?')
# red_patch = mpatches.Patch(color='red', label='True Values')
# blue_patch = mpatches.Patch(color='blue', label='Predicted Values')
# black_patch = mpatches.Patch(color='black', label='Correct Classification?')
# plt.legend(handles=[red_patch, blue_patch, black_patch])
# plt.show()

debug = True
