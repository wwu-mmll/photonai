import warnings

import numpy as np
from nilearn.datasets import fetch_oasis_vbm
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Preprocessing
from photonai.neuro import AtlasMapper, NeuroBranch

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Specify where the results should be written to and the name of your analysis
results_folder = './tmp/'
cache_folder = './tmp/cache'

# GET DATA FROM OASIS
n_subjects = 200
dataset_files = fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
gender = dataset_files.ext_vars['mf'].astype(str)
y = np.array(gender)
X = np.array(dataset_files.gray_matter_maps)


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
settings = OutputSettings(project_folder=results_folder)


# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('atlas_mapper_example',
                    optimizer='grid_search',
                    metrics=['accuracy'],
                    best_config_metric='accuracy',
                    inner_cv=KFold(n_splits=2),
                    verbosity=2,
                    output_settings=settings,
                    cache_folder=cache_folder)

preprocessing = Preprocessing()
preprocessing += PipelineElement("LabelEncoder")
my_pipe += preprocessing
my_pipe += PipelineElement('LinearSVC')


# DEFINE NEURO ELEMENTS
#brain_atlas = PipelineElement('BrainAtlas', atlas_name="Yeo_7", extract_mode='vec',
#                                rois='all', batch_size=200)
brain_atlas = PipelineElement('BrainAtlas', atlas_name="AAL",
                              rois=['Hippocampus_L', 'Hippocampus_R', "Frontal_Sup_Orb_L"], batch_size=200)

neuro_branch = NeuroBranch('NeuroBranch')
neuro_branch += brain_atlas

# NOW TRAIN ATLAS MAPPER
atlas_mapper = AtlasMapper(create_surface_plots=True)
atlas_mapper.generate_mappings(my_pipe, neuro_branch, results_folder)
atlas_mapper.fit(X, y)

# LOAD TRAINED ATLAS MAPPER AND PREDICT
atlas_mapper = AtlasMapper()
# you can either load an atlas mapper by specifying the atlas_mapper_meta.json file that has been created during fit()
# or simply specify the results folder in which your model was saved (and you can also specify the analysis name in case
# there are multiple atlas mapper within one folder)

#atlas_mapper.load_from_file(os.path.join(results_folder) + 'atlas_mapper_meta.json')
atlas_mapper.load_from_folder(folder=results_folder, analysis_name='atlas_mapper_example')
print(atlas_mapper.predict(X))
debug = True


