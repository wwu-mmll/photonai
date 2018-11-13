# -------------------- GENERATED WITH PHOTON WIZARD (beta) ------------------------------
# PHOTON Code File (Photon Syntax): 
# /spm-data/Scratch/photon_wizard/rleenings/bostonhousing/photon_code.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import Categorical, FloatRange

# Load data
df = pd.read_excel('/spm-data/Scratch/photon_wizard/rleenings/examples/boston_housing/boston_data.xlsx')
X = np.asarray(df.iloc[:, 0:13])
y = np.asarray(df.iloc[:, 13])
# covariates = data[:,None]

# Define cross-validation strategies
outer_cv = KFold(n_splits=5, shuffle=True)
inner_cv = KFold(n_splits=5, shuffle=True)

# Specify how results are going to be saved
persist_options = PersistOptions(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_results",
                                 save_predictions="best",
                                 save_feature_importances="None",
                                 local_file="/spm-data/Scratch/photon_wizard/rleenings/bostonhousing/photon_results.p",
                                 log_filename="/spm-data/Scratch/photon_wizard/rleenings/bostonhousing/photon_results.log",
                                 pretrained_model_filename="/spm-data/Scratch/photon_wizard/rleenings/bostonhousing/photon_model.photon",
                                 summary_filename="/spm-data/Scratch/photon_wizard/rleenings/bostonhousing/photon_summary.txt",
                                 user_id="rleenings",
                                 wizard_object_id="5bd83dd4a981127248205af3",
                                 wizard_project_name="bostonhousing")

# Define hyperpipe
hyperpipe = Hyperpipe('Boston Housing',
                      optimizer='grid_search', optimizer_params={},
                      metrics=['mean_absolute_error'],
                      best_config_metric='mean_absolute_error',
                      outer_cv=outer_cv,
                      inner_cv=inner_cv,
                      eval_final_performance=True,
                      verbosity=1,
                      persist_options=persist_options)

# Add transformer elements
hyperpipe += PipelineElement("StandardScaler", {'with_mean': [True], 'with_std': [True]}, test_disabled=True)
hyperpipe += PipelineElement("PCA", {'n_components': [None, 10], 'random_state': [15], 'whiten': [False]},
                             test_disabled=False)
# Add estimator
hyperpipe += PipelineElement("SVR", {'C': [0.5, 1.0, 1.5, 2.0],
                                     'epsilon': [0.1],
                                     'gamma': [],
                                     'kernel': ['linear', 'rbf']})

# Fit hyperpipe
hyperpipe.fit(X, y)

