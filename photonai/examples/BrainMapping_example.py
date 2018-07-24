from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineSwitch, PersistOptions
from photonai.neuro.AtlasStacker import AtlasInfo
from photonai.neuro.AtlasMapping import AtlasMapping
from sklearn.model_selection import KFold, ShuffleSplit, GroupKFold
from nilearn import datasets

# get oasis gm data and age from nilearn; imgs
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=90)
dataset_files = oasis_dataset.gray_matter_maps
targets = oasis_dataset.ext_vars['age'].astype(float)   # age

# where to write the results
folder = ''

# which atlases are available?
#BrainAtlas.whichAtlases()

# define hyperpipe to be applied to each ROI as usual
pers_opts = PersistOptions(local_file='dummy_file',
                           save_predictions='best',
                           save_feature_importances='None')
my_pipe = Hyperpipe(name='dummy_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error', 'mean_squared_error', 'pearson_correlation'],
                    best_config_metric='mean_absolute_error',
                    outer_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                    inner_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                    persist_options=pers_opts,
                    verbosity=0)
my_pipe += PipelineElement('StandardScaler')
my_pipe += PipelineElement('SVR', {'kernel': ['linear', 'rbf']})


# get info for the atlas
atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Precentral_L', 'Precentral_R', 'Frontal_Sup_R'], extraction_mode='vec')
#atlas_info = AtlasInfo(atlas_name='HarvardOxford-cort-maxprob-thr50', roi_names='all', extraction_mode='mean')
#atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='vec')
#atlas_info = AtlasInfo(atlas_name='AAL', roi_names=[2001, 2002, 2102], extraction_mode='vec')
#atlas_info = AtlasInfo(atlas_name='AAL', roi_names='all', extraction_mode='box')
# atlas_info = AtlasInfo(atlas_name='mni_icbm152_t1_tal_nlin_sym_09a_mask', mask_threshold=.5, roi_names='all', extraction_mode='vec')

# fit hyperpipe to every ROI independently and return the results
roi_results_table = AtlasMapping.mapAtlas(dataset_files=dataset_files, targets=targets, hyperpipe=my_pipe, atlas_info=atlas_info, write_to_folder=folder)

debug = True


