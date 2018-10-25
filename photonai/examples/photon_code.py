# Import all necessary python packages

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit

from photonai.investigator.Investigator import Investigator
from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PersistOptions
from photonai.optimization.Hyperparameters import Categorical, FloatRange

# Load data
df = pd.read_excel('/spm-data/Scratch/photon_wizard/rleenings/randomforestshit/features.xlsx')
X = np.asarray(df.iloc[:, 2:32])
y = np.asarray(df.iloc[:, 1])
# covariates = data[:,0]

# Define cross-validation strategies
outer_cv = KFold(n_splits=10, shuffle=True)
inner_cv = KFold(n_splits=5, shuffle=True)

# Specify how results are going to be saved
persist_options = PersistOptions(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_results",
                                 save_predictions="best",
                                 save_feature_importances="best",
                                 local_file="/spm-data/Scratch/photon_wizard/rleenings/randomforestshit/photon_results.p",
                                 log_filename="/spm-data/Scratch/photon_wizard/rleenings/randomforestshit/photon_results.log",
                                 user_id="rleenings",
                                 wizard_object_id="5bd08900a98112154bfa5c24",
                                 wizard_project_name="randomforestshit")

# Define hyperpipe
hyperpipe = Hyperpipe('Random Forest Shit',
                      optimizer='random_grid_search', optimizer_params={'k': 100},
                      metrics=['accuracy'],
                      best_config_metric='accuracy',
                      outer_cv=outer_cv,
                      inner_cv=inner_cv,
                      eval_final_performance=True,
                      verbosity=1,
                      persist_options=persist_options)

# Add transformer elements
hyperpipe += PipelineElement("StandardScaler", {}, test_disabled=False)
hyperpipe += PipelineElement("PCA", {'n_components': [5, 10, 20, 25, None]}, test_disabled=False)
# Add estimator
hyperpipe += PipelineElement("RandomForestClassifier", {'criterion': ['gini'],
                                                        'max_depth': [None],
                                                        'min_samples_leaf': [1],
                                                        'min_samples_split': [2],
                                                        'n_estimators': [10],
                                                        'n_jobs': [-1]}, test_disabled=False)

# Fit hyperpipe
hyperpipe.fit(X, y)

Investigator.load_from_db('mongodb://trap-umbriel:27017/photon_results', 'Random Forest Shit')
debug = True