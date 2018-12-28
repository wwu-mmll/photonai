        
# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Code File (Photon Syntax): 
# /spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/photon_code.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import Categorical, FloatRange

# Load data
df = pd.read_excel('/spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/breast_cancer_data.xlsx')
X = np.asarray(df.iloc[:,2:31])
y = np.asarray(df.iloc[:,1])
# covariates = data[:,None]
            
# Define cross-validation strategies
outer_cv = KFold(n_splits=3, shuffle=True)
inner_cv = KFold(n_splits=5, shuffle=True)

# Specify how results are going to be saved
persist_options = PersistOptions(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_results",
                                     save_predictions="best",
                                     save_feature_importances="None",
                                     local_file="/spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/photon_results.p",
                                     log_filename="/spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/photon_results.log",
                                     pretrained_model_filename="/spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/photon_model.photon",
                                     summary_filename="/spm-data/Scratch/photon_wizard/rleenings/breastcancerdemopipeline/photon_summary.txt",
                                     user_id="rleenings",
                                     wizard_object_id="5c1cc371a98112562d768cba",
                                     wizard_project_name="breastcancerdemopipeline")
                
# Define hyperpipe
hyperpipe = Hyperpipe('breastcancerdemopipeline',
                    optimizer='grid_search', optimizer_params={},
                    metrics=['accuracy', 'f1_score'],
                    best_config_metric='accuracy',
                    outer_cv=outer_cv,
                    inner_cv=inner_cv,
                    eval_final_performance=True,
                    verbosity=1,
                    persist_options=persist_options)
                
# Add transformer elements
hyperpipe += PipelineElement("StandardScaler", {'with_mean': [True], 'with_std': [True]}, test_disabled=False)
hyperpipe += PipelineElement("PCA", {'n_components': [5, 10, 15, 20], 'random_state': [15]}, test_disabled=False)
# Add estimator
hyperpipe += PipelineElement("SVC", {'C': [0.5, 1.0, 1.5, 2.0], 'kernel': ['rbf', 'linear']})

# Fit hyperpipe
hyperpipe.fit(X, y)
                                

# call PHOTON Investigator for a visualization of results 
from photonai.investigator.Investigator import Investigator
Investigator.show(hyperpipe)
        
        
