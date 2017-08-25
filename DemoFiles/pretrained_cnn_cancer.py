import glob
from pathlib import Path

import numpy as np
from PIL import Image
from progressbar import ProgressBar
from sklearn.model_selection import ShuffleSplit

import Helpers.TFUtilities as tfu
from Framework.PhotonBase import PipelineElement, Hyperpipe


##
# Loads the images. They have to be 299x299 pixels.
# returns (X_train, y_train)
def load_skin_cancer_data(use_tempfiles: True):
    path_benign_images='/spm-data-cached/Scratch/spielwiese_claas/ISIC-images-benign/299x299/'
    path_malignant_images='/spm-data-cached/Scratch/spielwiese_claas/ISIC-images-malignant/299x299/'
    path_temp_files_folder='/spm-data-cached/Scratch/spielwiese_claas/'

    skin_cancer_data = Path(path_temp_files_folder + "skin_cancer_data.npy")
    skin_cancer_labels = Path(path_temp_files_folder + "skin_cancer_labels.npy")


    # checking if temp-files already exists
    if use_tempfiles and skin_cancer_data.is_file() and skin_cancer_labels.is_file():
        print("Loading temp-files")
        X_train = np.load(skin_cancer_data)
        y_train = np.load(skin_cancer_labels)

    else:
        print("No temp-files! Loading image-Files...")
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

    print("loading done!")
    return (X_train, y_train)

def split_balanced_test_data_from_current_data(X_train, y_train):
     X_test = np.concatenate([X_train[:100],  X_train[-100:]])
     y_test = np.concatenate([y_train[:100],  y_train[-100:]])
     X_train_new = X_train[100:-100]
     y_train_new = y_train[100:-100]
     return {'X_train': X_train_new, 'y_train': y_train_new, 'X_test': X_test, 'y_test': y_test}

X_train, y_train = load_skin_cancer_data(use_tempfiles=True)

cv = ShuffleSplit(n_splits=1,test_size=0.2, random_state=23)


print("Size x_train: " + str(X_train.shape))
print("Size y_train: " + str(y_train.shape))
print("Sum benign: " + str(np.sum(y_train[:,0])))
print("Sum malignant: " + str(np.sum(y_train[:,1])))

#cv = KFold(n_splits=5, random_state=23)
my_pipe = Hyperpipe('Skin Cancer VGG18 finetuning', optimizer='grid_search',
                            metrics=['categorical_accuracy', 'f1_score', 'confusion_matrix'], best_config_metric='categorical_accuracy',
                            hyperparameter_specific_config_cv_object=cv,
                            hyperparameter_search_cv_object=cv,
                            eval_final_performance=True, verbose=2)
#my_pipe += PipelineElement.create('standard_scaler')
my_pipe += PipelineElement.create('PretrainedCNNClassifier', {'input_shape': [(299,299,3)],'target_dimension': [2], 'freezing_point':[249], 'batch_size':[32], 'early_stopping_flag':[True], 'eaSt_patience':[50]}, nb_epoch=1)
#my_pipe.fit(X_train, y_train)
#y_pred = my_pipe.predict(X_test)
#print(accuracy_score(one_hot_to_binary(y_test), y_pred))