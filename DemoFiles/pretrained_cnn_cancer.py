import glob
from pathlib import Path

import numpy as np
from PIL import Image
from progressbar import ProgressBar
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import accuracy_score
from keras.models import load_model

import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe
from Logging.Logger import Logger

logger = Logger()
development = False # If True using smaller samples
ROOT_PATH = '/spm-data-cached/Scratch/spielwiese_claas/'
##
# Loads the images. They have to be 299x299 pixels.
# returns (X_train, y_train)
def load_skin_cancer_data(use_tempfiles: True, load_fraction: 1.0):
    root_path = ROOT_PATH
    #root_path = '/home/claas/skin_cancer/'
    path_benign_images=root_path + 'ISIC-images-benign/299x299/'
    path_malignant_images=root_path + 'ISIC-images-malignant/299x299/'
    path_temp_files_folder=root_path + ''

    skin_cancer_data = Path(path_temp_files_folder + "skin_cancer_data.npy")
    skin_cancer_labels = Path(path_temp_files_folder + "skin_cancer_labels.npy")


    # checking if temp-files already exists
    if use_tempfiles and skin_cancer_data.is_file() and skin_cancer_labels.is_file():
        logger.info("Loading temp-files")
        X_train = np.load(skin_cancer_data)
        y_train = np.load(skin_cancer_labels)
    else:
        logger.info("No temp-files! Loading image-Files...")
        data = []
        labels = []

        # Loading benign images
        pbarb = ProgressBar()
        labels_benign = []
        data_benign = []
        for infile in pbarb(glob.glob(path_benign_images + "*")):
             im = Image.open(infile)
             im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
             im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
             data_benign.append(im_arr)
             labels_benign.append([1,0])

        # Loading malignant images
        labels_malignant = []
        data_malignant = []
        pbarm = ProgressBar()
        for infile in pbarm(glob.glob(path_malignant_images + "*")):
            im = Image.open(infile)
            im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
            data_malignant.append(im_arr)
            labels_malignant.append([0,1])

        amount_benign = int(len(labels_benign) * load_fraction)
        amount_malignant = int(len(labels_malignant) * load_fraction)
        data = data_benign[:amount_benign] + data_malignant[:amount_malignant]
        labels = labels_benign[:amount_benign] + labels_malignant[:amount_malignant]

        X_train = np.asarray(data)
        y_train = np.asarray(labels)

        if use_tempfiles:
            np.save(str(skin_cancer_data), X_train)
            np.save(str(skin_cancer_labels), y_train)

    logger.info("loading done!")
    return (X_train, y_train)

def split_balanced_test_data_from_current_data(X_train, y_train, group_size=100):
    logger.info("Extract independent balanced test sample. {} per class.".format(group_size))
    X_test = np.concatenate([X_train[:group_size],  X_train[-group_size:]])
    y_test = np.concatenate([y_train[:group_size],  y_train[-group_size:]])
    X_train_new = X_train[group_size:-group_size]
    y_train_new = y_train[group_size:-group_size]
    if development:
        logger.warn("!!!! Development mode is enabled, using smaller samples !!!!")
        X_train_new = X_train_new[0:500]
        y_train_new = y_train_new[0:500]
    return (X_train_new, y_train_new, X_test, y_test)

def save_original_vs_predict_csv(filename, y, y_predicted):
    import csv
    y_predicted_list = np.array(y_predicted)
    with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(0, len(y)):
                row = []
                if len(y[i] == 2):
                    row.append(y[i][0])
                    row.append(y[i][1])
                else:
                    row.append(y[i])
                if len(y_predicted_list) == 2:
                    row.append(y_predicted_list[i][0])
                    row.append(y_predicted_list[i][1])
                else:
                    row.append(y_predicted_list[i])
                csv_writer.writerow(row)

X_train, y_train = load_skin_cancer_data(use_tempfiles=False, load_fraction=1.0)
X_train, y_train, X_test, y_test = split_balanced_test_data_from_current_data(X_train, y_train)

cv = ShuffleSplit(n_splits=1,test_size=0.2, random_state=23)

logger.info(
    """
    Size x_train:  {0}
    Size y_train:  {1}
    Sum benign: {2}
    Sum malignant: {3}
    """.format(str(X_train.shape), str(y_train.shape), str(np.sum(y_train[:,0])), str(np.sum(y_train[:,1])))
)


#cv = KFold(n_splits=5, random_state=23)
my_pipe = Hyperpipe('Skin Cancer VGG18 finetuning', optimizer='grid_search',
                    metrics=['categorical_accuracy', 'f1_score', 'confusion_matrix', 'accuracy'], best_config_metric='categorical_accuracy',
                    inner_cv=cv,
                    outer_cv=cv,
                    eval_final_performance=True, verbose=2)
#my_pipe += PipelineElement.create('standard_scaler')

weight_benign = 1
weight_malignant = np.sum(y_train[:,0]) / np.sum(y_train[:,1])

my_pipe += PipelineElement.create('PretrainedCNNClassifier',
                                  {'input_shape': [(299,299,3)],'target_dimension': [2],
                                   'freezing_point':[249], 'batch_size':[16],
                                   'early_stopping_flag':[False], 'eaSt_patience':[5], 'weight_class_a':[weight_benign], 'weight_class_b':[weight_malignant]},
                                  nb_epoch=15, ckpt_name='{0}{1}'.format(ROOT_PATH, 'pretrained_cnn_cancer.hdf5'))
my_pipe.fit(X_train, y_train)
y_pred = my_pipe.predict(X_test)

# RESULTS
result_tree = my_pipe.result_tree

from Framework import LogExtractor
log_ex = LogExtractor.LogExtractor(result_tree)
import datetime
import time
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M')
log_ex.extract_csv("{0}{1}_{2}.csv".format(ROOT_PATH, "skin_cancer_results", st))
save_original_vs_predict_csv("{0}{1}_{2}.csv".format(ROOT_PATH, "skin_cancer_y_vs_y-predicted", st), y_test, y_pred)

balanced_accuracy = accuracy_score(tfu.one_hot_to_binary(y_test), y_pred)
logger.info("Accuracy from independent balanced sample: {}".format(balanced_accuracy))
best_model = load_model("{0}{1}".format(ROOT_PATH, 'pretrained_cnn_cancer.hdf5'))
y_pred_best = best_model.predict(X_test)

save_original_vs_predict_csv("{0}{1}_{2}.csv".format(ROOT_PATH, "skin_cancer_y_vs_y-predicted_best", st), y_test, y_pred_best)

#logger.debug("y_pred_best: {}".format(y_pred_best))
y_pred_best = np.argmax(y_pred_best, axis=1)
balanced_accuracy_best = accuracy_score(tfu.one_hot_to_binary(y_test), y_pred_best)



logger.info("Accuracy for best model from independent balanced sample: {}".format(balanced_accuracy_best))
