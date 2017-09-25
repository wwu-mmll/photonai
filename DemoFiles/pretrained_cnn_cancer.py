import glob
from pathlib import Path

import numpy as np
from PIL import Image
from progressbar import ProgressBar
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from keras.models import load_model

import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe
from Logging.Logger import Logger

logger = Logger()
development = False # If True using smaller samples
##
# Loads the images. They have to be 299x299 pixels.
# returns (X_train, y_train)
def load_skin_cancer_data(use_tempfiles: True):
    root_path='/spm-data-cached/Scratch/spielwiese_claas/'
    root_path = '/home/claas/skin_cancer/'
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
        for infile in pbarb(glob.glob(path_benign_images + "*")):
             im = Image.open(infile)
             im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
             im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
             data.append(im_arr)
             labels.append(0)

        # Loading malignant images
        pbarm = ProgressBar()
        for infile in pbarm(glob.glob(path_malignant_images + "*")):
            im = Image.open(infile)
            im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
            data.append(im_arr)
            labels.append(1)

        X_train = np.asarray(data)
        y_train = np.asarray(labels)
        y_train = tfu.binary_to_one_hot(y_train)

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
        X_train_new = X_train_new[0:1000]
        y_train_new = y_train_new[0:1000]
    return (X_train_new, y_train_new, X_test, y_test)

X_train, y_train = load_skin_cancer_data(use_tempfiles=True)
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

ratio_true_false = np.count_nonzero(y_train)/np.size(y_train)
my_pipe += PipelineElement.create('PretrainedCNNClassifier',
                                  {'input_shape': [(299,299,3)],'target_dimension': [2],
                                   'freezing_point':[249], 'batch_size':[32],
                                   'early_stopping_flag':[True], 'eaSt_patience':[5], 'weight_class_b':[ratio_true_false]},
                                  nb_epoch=10, ckpt_name='/home/claas/pretrained_cnn_cancer.hdf5')
my_pipe.fit(X_train, y_train)
y_pred = my_pipe.predict(X_test)

# RESULTS
result_tree = my_pipe.result_tree

from Framework import LogExtractor
log_ex = LogExtractor.LogExtractor(result_tree)
log_ex.extract_csv("skin_cancer_results.csv")

balanced_accuracy = accuracy_score(tfu.one_hot_to_binary(y_test), y_pred)
logger.info("Accuracy from independent balanced sample: {}".format(balanced_accuracy))
best_model = load_model('pretrained_cnn_cancer.hdf5')
y_pred_best = best_model.predict(X_test)
logger.debug("y_pred_best: {}".format(y_pred_best))
y_pred_best = np.argmax(y_pred_best, axis=1)
balanced_accuracy_best = accuracy_score(tfu.one_hot_to_binary(y_test), y_pred_best)
logger.info("Accuracy for best model from independent balanced sample: {}".format(balanced_accuracy_best))
