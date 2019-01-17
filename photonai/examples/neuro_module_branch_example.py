"""
PHOTON Example of NeuroModuleBranch

In this example, we take a look at the NeuroModuleBranch class that can be used just like a PHOTON PipelineBranch, only
that it is designed to work for neuro PipelineElements only. We will build a neuro branch and add different neuro
PipelineElements to it. We will then try to predict the age of the subjects we look at. The data comes from the OASIS
dataset.

"""
from photonai.base.PhotonBase import Hyperpipe, PipelineElement
from photonai.neuro.NeuroBase import NeuroModuleBranch
from photonai.neuro.AtlasStacker import AtlasInfo
from nilearn import datasets
from sklearn.model_selection import KFold

# Get data first, 50 subjects from the OASIS database, VBM images and age
oasis = datasets.fetch_oasis_vbm(n_subjects=50)
files = oasis.gray_matter_maps
targets = oasis.ext_vars['age'].astype(float)

# We start by building a NeuroModuleBranch including a brain atlas
neuro_branch = NeuroModuleBranch('NeuroBranch')
neuro_branch += PipelineElement('ResampleImgs', {'voxel_size': [[5, 5, 5]]})
atlas_info = AtlasInfo(atlas_name='AAL', roi_names=['Precentral_R'], extraction_mode='vec')
neuro_branch += PipelineElement('BrainAtlas', {}, atlas_info_object=atlas_info)

# Now, we build a Hyperpipe and add the neuro branch to it
pipe = Hyperpipe('neuro_module_branch_example', optimizer='grid_search',
                    optimizer_params={},
                    metrics=['mean_squared_error', 'mean_absolute_error'],
                    best_config_metric='mean_squared_error',
                    inner_cv=KFold(n_splits=2, shuffle=True, random_state=3),
                    outer_cv=KFold(n_splits=2, shuffle=True, random_state=3),
                    eval_final_performance=True)
pipe += neuro_branch

# Finally, we add an estimator
pipe += PipelineElement('SVR', {}, kernel='linear', C=0.001)

# We can now run PHOTON and try to predict age
pipe.fit(files, targets)
