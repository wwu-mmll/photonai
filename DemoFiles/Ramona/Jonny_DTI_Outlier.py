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

def load_covariates(xls_file_path: str):
    age = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="E", squeeze=True)
    gender = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="F", squeeze=True)
    hamilton_pre = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="G", squeeze=True)
    # icv = pandas.read_excel(open(xls_file_path, 'rb'), sheet_name='ECT', usecols="F", squeeze=True)
    return (age, gender, hamilton_pre)

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
    cached_preprocessed_dti_data_file = Path(ROOT_DIR + "/ECT_data_for_Tim/cached_preprocessed_dti_data_fa.npy")
    if cached_preprocessed_dti_data_file.exists():
        Logger().info("Loading cached preprocessed DTI data...")
        dti_roi_brain_striped = np.load(cached_preprocessed_dti_data_file)
    else:
        dti_image_files = load_dti_images(ROOT_DIR + '/ECT_data_for_Tim/FA/', subject_ids)
        dti_roi_brain = extract_brain_from_dti_images(dti_image_files)
        dti_roi_brain_striped = extract_unique_dti_features(dti_roi_brain)
        dti_roi_brain_striped = np.nan_to_num(dti_roi_brain_striped)
        np.save(cached_preprocessed_dti_data_file, dti_roi_brain_striped)
    return dti_roi_brain_striped


class CustomDTISplit:

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):
        test = list(np.where(y == 1))
        train = list(np.where(y == 0))
        yield (train, test)


def classical_classification() -> Hyperpipe:

    pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                     metrics=['accuracy'],
                     best_config_metric='accuracy',
                     inner_cv=CustomDTISplit(),
                     # outer_cv=ShuffleSplit(),
                     # inner_cv=ShuffleSplit(),
                     eval_final_performance=False)

    # hyperparameters={'p_threshold': [0.001, 0.01, 0.05]}
    # pipe += PipelineElement.create("pearson_feature_selector",
    #                                test_disabled=False, p_threshold=0.001)
    pipe += PipelineElement.create("standard_scaler", hyperparameters={}, test_disabled=False)
    pipe += PipelineElement.create("pca", hyperparameters={}, test_disabled=False, n_components=None)
    pipe += PipelineElement.create("OneClassSVM", hyperparameters={'nu': [0.2, 0.4, 0.5, 0.7, 0.9]})

    return pipe


def fit_model(targets, age, gender, hamilton_pre):
    # Load Data and create pipeline
    dti_preprocessed = load_and_preprocess_dti_data()
    pipe = classical_classification()

    features = np.ascontiguousarray(np.array(np.squeeze(dti_preprocessed)))

    features = np.hstack((features, np.reshape(age.values, (-1, 1))))
    features = np.hstack((features, np.reshape(gender.values, (-1, 1))))
    features = np.hstack((features, np.reshape(hamilton_pre.values, (-1, 1))))

    targets_binary = np.zeros(targets.shape)
    targets_indices = np.where(targets < 7)
    targets_binary[targets_indices] = 1
    targets_binary = [int(i) for i in targets_binary]

    Logger().info('Starting Hyperparameter Search')
    pipe.fit(features, targets_binary)

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
            mse = fold.test.metrics['accuracy']
            print('accuracy: ' + str(mse))

        pickle.dump(pearson_list_y_true, open('jonny_pipe_y_true.p', 'wb'))
        pickle.dump(pearson_list_y_pred, open('jonny_pipe_y_pred.p', 'wb'))

    # logex = LogExtractor.LogExtractor(pipe.result_tree)
    # logex.extract_csv('johnny_dti_results.csv')


def evaluate_predictions(xls_file, targets):
    pearson_list_y_true = pickle.load(open('jonny_pipe_y_true.p', 'rb'))#
    pearson_list_y_pred = pickle.load(open('jonny_pipe_y_pred.p', 'rb'))


xls_file = ROOT_DIR + '/Key_Information_ECT_sample_20170829.xlsx'
subject_ids, targets = load_etc_subject_ids_and_targets(xls_file)
age, gender, hamilton_pre = load_covariates(xls_file)
fit_model(targets, age, gender, hamilton_pre)
# evaluate_predictions(xls_file, targets)

# plot_predictions()

# plt.plot(sorted(pearson_list_y_true))
# plt.plot(np.ones(len(pearson_list_y_true), ) * 7)
# plt.show()


debug = True
