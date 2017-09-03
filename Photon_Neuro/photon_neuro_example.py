from Framework.PhotonBase import PipelineElement, Hyperpipe
from sklearn.model_selection import KFold

# get oasis gm data and age from nilearn
from nilearn import datasets
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=10)
dataset_files = oasis_dataset.gray_matter_maps
targets = oasis_dataset.ext_vars['age'].astype(float) # age

# setup photon HP
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                    optimizer_params={},
                    #metrics=['accuracy', 'precision', 'f1_score'],
                    hyperparameter_specific_config_cv_object=KFold(n_splits=2, random_state=3),
                    hyperparameter_search_cv_object=KFold(n_splits=3, random_state=3),
                    eval_final_performance=False)

my_pipe += PipelineElement.create('SmoothImgs', {'fwhr': [[8, 8, 8], [12, 12, 12]], 'output_img': [False]})
#my_pipe += PipelineElement.create('ResampeImgs', {'voxel_size': [[5, 5, 5], [10, 10, 10]], 'output_img': [False]})
my_pipe += PipelineElement.create('SVR', {'kernel': ['linear', 'rbf']})

# START HYPERPARAMETER SEARCH
my_pipe.fit(dataset_files, targets)


# resImg = ResamplingImgs(voxel_size=[5, 5, 5], output_img=True).transform(dataset_files)
# smImg = SmoothImgs(fwhr=[6, 6, 6], output_img=True).transform(resImg)
#
# from Photon_Neuro0.BrainAtlas import BrainAtlas
# myAtlas = BrainAtlas(atlas_name='AAL',
#                      extract_mode='vec',
#                      whichROIs='all')
# roi_data = myAtlas.transform(X=smImg)
# myAtlas.getInfo()

print('')