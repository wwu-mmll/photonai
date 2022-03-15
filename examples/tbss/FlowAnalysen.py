import pandas as pd
import numpy as np
import os
import glob
from photonai.base import Hyperpipe, PipelineElement, OutputSettings, Switch, Preprocessing, CallbackElement
from photonai.optimization import Categorical, IntegerRange, FloatRange
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as bac

# requires glmnet: conda install -c conda-forge glmnet

# Datei Flow309: Multiclass mit Standardpipeline + LASSO --> preds und feature importances
# Datei OLINK158: Multiclass mit Standardpipeline + LASSO --> preds und feature importances
# Datei Flow158: Multiclass mit Standardpipeline + LASSO --> preds und feature importances
# Datei OLINK158 und Flow158 zusammen: Multiclass mit Standardpipeline + LASSO --> preds und feature importances
# opt lambda for LASSO


# Load data
# pref = "C:/Users/Tim/Google Drive/work/all_skripts/py_code/Neurologie/MegaPhenotype/"
#files = glob.glob(pref + 'data/FlowAnalysen/' + '*.{}'.format("xlsx"))
# files = ['C:/Users/Tim/Google Drive/work/all_skripts/py_code/Neurologie/MegaPhenotype/data/FlowAnalysen/Olink_158.xlsx']
pref = "./"
files = ['./Olink_158.csv']

# get features
for f in files:
    print('--------------------------------------------------------------------------------------------------------')
    analysis_name = os.path.basename(f)[:-5]
    print('File: ' + f)
    # get data
    # df = pd.read_excel(f)
    df = pd.read_csv(f, sep=";")
    df_y = df.iloc[:, 0]
    df_X = df.drop(df.columns[0], axis=1)

    # drop -99
    out_ind = np.squeeze(df_y.values == -99)

    y = np.squeeze(df_y.values[~out_ind]).astype(int) - 1
    X = df_X.values[~out_ind]

    # def c(arr):
    #     convertArr = []
    #     for s in arr.ravel():
    #         try:
    #             value = np.float32(s)
    #         except ValueError:
    #             value = np.nan
    #             print(s)
    #         convertArr.append(value)
    #     return np.array(convertArr,dtype=object).reshape(arr.shape)
    #
    # X = c(X).astype(float)

    #from PHOTON_permutation_importance import permutation_importance
    #permutation_importance(estimator, X, y, scoring=None, n_repeats=5, random_state=42, sample_weight=None)

    def my_monitor(X, y=None, **kwargs):
        print(X.shape)
        debug = True

    # Define cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = [("bac", bac)]   #, 'accuracy', "matthews_corrcoef"]
    bcm = "bac"

    # Define my_pipe
    outputsettings = OutputSettings(overwrite_results=True)
    my_pipe = Hyperpipe(name='Neurologie_' + analysis_name, optimizer='grid_search',
                        metrics=metrics, best_config_metric=bcm,
                        inner_cv=inner_cv,
                        #outer_cv=outer_cv,
                        use_test_set=False,
                        verbosity=1, cache_folder=None, output_settings=outputsettings,
                        project_folder=pref + 'results/FlowAnalysen/analysis_' + analysis_name)

    # Add transformer elements
    #my_pipe += PipelineElement("SimpleImputer")   #, test_disabled=False, missing_values=np.nan, strategy='mean', fill_value=0)
    my_pipe += PipelineElement("RobustScaler")
    #my_pipe += CallbackElement("monitor", my_monitor)
    #my_pipe += PipelineElement("FClassifSelectPercentile",  hyperparameters={'percentile': [5, 10, 30, 50, 90]},
    #                           test_disabled=True)

    # my_pipe += PipelineElement("ImbalancedDataTransformer",
    #                              #hyperparameters={'method_name': Categorical(['RandomUnderSampler', 'SMOTE'])},
    #                              hyperparameters={'method_name': Categorical(['SMOTE'])},
    #                              test_disabled=False)

    my_pipe += PipelineElement("ImbalancedDataTransformer",  # hyperparameters={'method_name': Categorical(['SMOTE'])},
                               method_name='SMOTE',
                               test_disabled=False)

    #estimator_switch = Switch('EstimatorSwitch')
    #estimator_switch += PipelineElement("RandomForestClassifier")

    from sklearn.ensemble import RandomForestClassifier
    #my_pipe += PipelineElement("AdaBoostClassifier", hyperparameters={'base_estimator': RandomForestClassifier()})
    my_pipe += PipelineElement("RandomForestClassifier")
    #my_pipe += estimator_switch

    # Fit my_pipe
    my_pipe.fit(X, y)

    print('Feature Importances')
    r = my_pipe.get_permutation_feature_importances(n_repeats=5, random_state=0)
    np.savetxt('results/nachanalysen/analysis_' + analysis_name + '/FeatureImportance_Neurologie' + analysis_name + '.csv',
               r["mean"], delimiter=",")
    np.savetxt('results/nachanalysen/analysis_' + analysis_name + '/FeatureImportanceStd_Neurologie_' + analysis_name + '.csv',
               r["std"], delimiter=",")

debug = True
