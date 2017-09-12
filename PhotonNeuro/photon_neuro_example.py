from Framework.PhotonBase import PipelineElement, Hyperpipe
from PhotonNeuro.BrainAtlas import  BrainAtlas
from PhotonNeuro.AtlasStacker import AtlasStacker, AtlasInfo
from sklearn.model_selection import KFold

# get oasis gm data and age from nilearn
# imgs
from nilearn import datasets
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=20)
dataset_files = oasis_dataset.gray_matter_maps
targets = oasis_dataset.ext_vars['age'].astype(float)   # age

# # data
# from sklearn.datasets import load_breast_cancer
# dataset = load_breast_cancer()
# dataset_files = dataset.data
# targets = dataset.target

print(BrainAtlas._getAtlasDict())

# setup photon HP
my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error', 'mean_absolute_error'],
                    inner_cv=KFold(n_splits=2, shuffle=True, random_state=3),
                    outer_cv=KFold(n_splits=2, shuffle=True, random_state=3),
                    eval_final_performance=True)

my_pipe += PipelineElement.create('SmoothImgs', {'fwhr': [[8, 8, 8], [12, 12, 12]]})
my_pipe += PipelineElement.create('ResampleImgs', {'voxel_size': [[5, 5, 5]]})

atlas_info = AtlasInfo(atlas_name='mni_icbm152_t1_tal_nlin_sym_09a_mask', mask_threshold=.5,
                       roi_names='all', extraction_mode='vec')
#atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='box')
my_pipe += PipelineElement.create('BrainAtlas', {}, atlas_info_object=atlas_info)
# my_pipe += PipelineElement('atlas_stacker',
#                            AtlasStacker(atlas_info, [['SVR', {'kernel': ['rbf', 'linear']}, {}]]),
#                            {})

my_pipe += PipelineElement.create('SVR', {'kernel': ['linear']})

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