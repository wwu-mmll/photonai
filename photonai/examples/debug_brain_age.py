import numpy as np
from nilearn.image import load_img
from sklearn.model_selection import KFold
from photonai.base import PipelineElement, Hyperpipe

from photonai.neuro import NeuroBranch
import pandas as pd
import glob

#load learn files with labels
gm_list_learn = pd.read_csv("/spm-data/Scratch/spielwiese_vincent/brain_age_CTQ_project_070119_all_studies.csv")
nifti_learn_paths = gm_list_learn['vbm_gray_matter_path']
age_to_learn = gm_list_learn['age']
# print('loading nifti files')
# nifti_learn = load_img(nifti_learn_paths).get_data()
# print(nifti_learn.shape)
# nifti_learn = np.moveaxis(nifti_learn, 3, 0)
# print(nifti_learn.shape)


# set up pipeline
my_pipe = Hyperpipe('MS_Brain_Age_Pipe',
                    optimizer='grid_search',
                    metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_absolute_error',
                    inner_cv=KFold(n_splits=2, shuffle=True, random_state=42),
                    outer_cv=KFold(n_splits=2, shuffle=True, random_state=42),
                    verbosity=2,
                    eval_final_performance=True)

mask = PipelineElement('BrainMask', mask_image='MNI_ICBM152_GrayMatter',
                       extract_mode='vec', batch_size=10)

neuro_branch = NeuroBranch('NeuroBranch', nr_of_processes=1)
neuro_branch += mask

my_pipe += neuro_branch

my_pipe += PipelineElement('SVR')


nifti_learn_paths = nifti_learn_paths[:20]
age_to_learn = age_to_learn[:20]

#train pipeline
my_pipe.fit(nifti_learn_paths, age_to_learn)

#load files to predict
#load predict files with labels
#load the files according to thegwm files
gmlist = glob.glob("/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/nifti/*/mri/mwp1nifti.nii")
#get RMS data
ID_excel1 = pd.read_excel("/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/Subjects_rms_processed.xlsx")
ID1 = ID_excel1['nifti_t1_dataset_id']
Predict_Age1 = ID_excel1.iloc[:, 3]
#get PMS data
ID_excel2 = pd.read_excel("/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/Subjects_pms_processed.xlsx")
ID2 = ID_excel2['nifti_t1_dataset_id']
Predict_Age2 = ID_excel2.iloc[:, 3]
#create empty file path list
file_path_list1 = []
file_path_list2 = []

for i in ID1:
    search_term = i
    file_path = '/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/nifti/' + search_term + '/mri/mwp2nifti.nii'
    file_path_list1.append(file_path)

for i in ID2:
    search_term = i
    file_path = '/spm-data/Scratch/spielwiese_vincent/MS_Prediction_Cat12/nifti/' + search_term + '/mri/mwp2nifti.nii'
    file_path_list2.append(file_path)

#get RMS niftis
X1 = load_img(file_path_list1).get_data()
X1 = np.moveaxis(X1, 3, 0)
X1 = np.reshape(X1, (X1.shape[0], -1))
#get PMS niftis
X2 = load_img(file_path_list2).get_data()
X2 = np.moveaxis(X2, 3, 0)
X2 = np.reshape(X2, (X2.shape[0], -1))

predict_niftis = np.concatenate((X1, X2), axis = 0)
print(predict_niftis.shape)
predict_age = np.concatenate((Predict_Age1, Predict_Age2), axis = 0)
print(predict_age.shape)

#predict with pipeline, save results and write them somwhere
csv_to_print = my_pipe.predict(predict_niftis)

