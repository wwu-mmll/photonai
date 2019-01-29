import pandas as pd
import numpy as np
from sklearn.model_selection import *

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import Categorical, IntegerRange, FloatRange
from photonai.validation.PermutationTest import PermutationTest

group_var = None


def create_hyperpipe():
    # Define cross-validation strategies
    outer_cv = KFold(n_splits=5, shuffle=True)
    inner_cv = KFold(n_splits=5, shuffle=True)

    # Specify how results are going to be saved
    output_settings = OutputSettings(mongodb_connect_url="mongodb://trap-umbriel:27017/photon_results",
                                     save_predictions="best",
                                     save_feature_importances="None",
                                     project_folder="/spm-data/Scratch/photon_wizard/nopel/enigmasvm5fold",
                                     user_id="nopel",
                                     wizard_object_id="5c4eda03720f60a48b9f894c",
                                     wizard_project_name="enigmasvm5fold")

    # Define hyperpipe
    hyperpipe = Hyperpipe('enigmasvm5fold',
                          optimizer='sk_opt', optimizer_params={},
                          metrics=['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc'],
                          best_config_metric='f1_score',
                          outer_cv=outer_cv,
                          inner_cv=inner_cv,
                          eval_final_performance=True,
                          verbosity=1,
                          output_settings=output_settings,
                          groups=group_var)

    # Add transformer elements
    hyperpipe += PipelineElement("SimpleImputer", hyperparameters={},
                                 test_disabled=False, missing_values=np.nan, strategy='mean', fill_value=0)
    hyperpipe += PipelineElement("StandardScaler", hyperparameters={},
                                 test_disabled=False, with_mean=True, with_std=True)
    hyperpipe += PipelineElement("PCA", hyperparameters={},
                                 test_disabled=False, n_components=None)
    hyperpipe += PipelineElement("ImbalancedDataTransform", hyperparameters={},
                                 test_disabled=False, method_name='RandomUnderSampler')
    # Add estimator
    hyperpipe += PipelineElement("SVC", hyperparameters={'C': FloatRange(0.5, 2)}, gamma='scale', kernel='linear')

    return hyperpipe

# Load data
df = pd.read_excel('/spm-data/Scratch/photon_wizard/nopel/enigmasvm5fold/BMIGr.xlsx')
X = np.asarray(df.iloc[:, 1::])
y = np.asarray(df.iloc[:, 0])

perm_tester = PermutationTest(create_hyperpipe, n_perms=20, n_processes=3, random_state=11)
perm_tester.fit(X, y)


