import pandas
import numpy as np


def get_data(pre, one_hot_it=False, what='all'):

    # get data (snps; recode SNPs to numerical values; 0-->hetero)
    snp_file = pre + 'Genetics/FORMarburgh_1KG_imputed_MDD_Compound_Genotypes_withSNPids_cleared.xlsx'
    snps_tmp, snp_names = get_snps(file=snp_file)
    snp_num_frame = recode_snps(snps_tmp, snp_names)  # recode SNP strings to numerical values

    if one_hot_it:
        snp_num_frame, snp_names = one_hot_snps(snp_num_frame, snp_names)  # one-hot encode snp matrix

    # get covariates (e.g. diagnosis)
    cov_file = pre + 'Datenbank_Update_DataFreeze1&2_17-11-2017_relVars.csv'
    # cov_file = pre + 'Datenbank_Update_DataFreeze1&2_01-12-2017_relVars2.csv'
    covs_tmp = get_covs(file=cov_file)

    # get targets
    target_tmp = get_targets(pre, what)

    # # drop NaNs from targets
    # impute_targets = 'drop'
    # # to keep train and test fully independent, always use drop
    # if impute_targets == 'drop':
    #     print('\nNaN-handling: drop')
    #     target_tmp = target_tmp.dropna(axis=0, how='any')
    # elif impute_targets == 'mean':
    #     print('\nNaN-handling: impute with mean')
    #     target_tmp = target_tmp.apply(lambda x: x.fillna(x.mean()), axis=0)

    # ToDo: drop duplicate cols

    ROI_names = list(target_tmp.columns[1:].values)  # skip subID and take the rest
    ###############################################################################################################

    # merge the three dataframes into one dataframe and only keep the intersection (via 'inner')
    df = pandas.merge(covs_tmp, target_tmp, how='inner', on='ID')
    df = pandas.merge(df, snp_num_frame, how='inner', on='ID')

    print('\nTarget shape: ' + str(target_tmp.shape))
    print('Covs shape: ' + str(covs_tmp.shape))
    print('SNPs shape: ' + str(snps_tmp.shape))
    print('Intersection merge: ' + str(df.shape))
    return df, ROI_names, snp_names

# get Genetic data
def get_snps(file):
    print('\nRetrieving SNPs...')
    snp_frame = pandas.read_excel(file)                    # read snp data
    snp_frame = snp_frame.dropna(axis=0, how='any')     # get rid of subjects whose data contains NaNs
    snp_names = snp_frame.columns[1:].values            # skip subID and take the rest
    snp_frame['ID'] = snp_frame['ID'].astype(str)       # cast ID to string
    return snp_frame, snp_names

# get clinical/psychometric etc. variables
def get_covs(file):
    print('\nRetrieving covariates...')
    covs_frame = pandas.read_csv(file)                 # read cov data
    covs_frame.columns = ['ID' if x == 'Proband' else x for x in covs_frame.columns]    # rename the ID column to allow merge
    covs_frame = covs_frame.dropna(axis=0, how='any', subset=['Group'])         # get rid of subjects whose data contains NaNs in the listed col(s) (we only need Group to be valid for everyone)
    covs_frame['ID'] = covs_frame['ID'].astype(str)
    subjID_covs= [s[0:4].lstrip("0") for s in covs_frame['ID']]  # only the first 5 characters are relevant, also remove leading zeros
    covs_frame = covs_frame.assign(ID=subjID_covs)
    return covs_frame

# get targets (ROI-wise cortical thickness or volumes or ...)
def get_targets_tmp(file):
    print('\nRetrieving targets...')
    target_frame = pandas.read_csv(open(file, 'rb'))              # read target data (e.g. volume or thickness)
    target_frame.columns = ['ID' if (x == 'SubjID' or x == 'names') else x for x in target_frame.columns] # rename the ID column to allow merge

    # only use first measurement (exclude the same subjects)
    cond = target_frame.ID.str.contains("-2_")
    target_frame = target_frame[~cond]

    # only the first 5 characters are relevant, also remove leading zeros
    subjID_target = [s[0:4].lstrip("0") for s in target_frame['ID']]
    target_frame = target_frame.assign(ID=subjID_target)
    target_frame['ID'] = target_frame['ID'].astype(str)

    #don't drop NaNs now, but target-wise in the loop to have max number of samples for each ROI model

    return target_frame

def get_targets(pre, what):
    # get targets (e.g. volumes, thickness, ... ); (CAT12_GM, DTI, rs-fMRI_Hubness, ...)
    if what[0] == 'all':
        what = ['thick', 'surf', 'vol', 'cat']

    # Thickness
    if any("thick" in s for s in what):
        print('\nRetrieving Freesurfer Thickness data...')
        target_file = pre + 'FreeSurfer_ROI/CorticalMeasuresENIGMA_ThickAvg.csv'
        target_thick = get_targets_tmp(file=target_file)
        if 'target_tmp' in locals():
            target_tmp = pandas.merge(target_tmp, target_thick, how='inner', on='ID')   # merge target datasets
        else:
            target_tmp = target_thick

    # SurfaceArea
    if any("surf" in s for s in what):
        print('\nRetrieving Freesurfer Surface data...')
        target_file = pre + 'FreeSurfer_ROI/CorticalMeasuresENIGMA_SurfAvg.csv'
        target_surf = get_targets_tmp(file=target_file)
        if 'target_tmp' in locals():
            target_tmp = pandas.merge(target_tmp, target_surf, how='inner', on='ID')    # merge target datasets
        else:
            target_tmp = target_surf

    # Volume
    if any("vol" in s for s in what):
        print('\nRetrieving Freesurfer Volume data...')
        target_file = pre + 'FreeSurfer_ROI/LandRvolumes.csv'
        target_vol = get_targets_tmp(file=target_file)
        if 'target_tmp' in locals():
            target_tmp = pandas.merge(target_tmp, target_vol, how='inner', on='ID')     # merge target datasets
        else:
            target_tmp = target_vol

    # CAT12 Vgm
    if any("cat" in s for s in what):
        print('\nRetrieving CAT12 ROI data...')
        target_file = pre + 'CAT12_ROI/ROI_CAT12_r1184_catROI_neuromorphometrics_Vgm.csv'
        target_Vgm = get_targets_tmp(file=target_file)
        if 'target_tmp' in locals():
            target_tmp = pandas.merge(target_tmp, target_Vgm, how='inner', on='ID')     # merge target datasets
        else:
            target_tmp = target_Vgm

    # handle custom targets string search
    if what[0] == 'custom_str':
        print('custom_str')
        tmp = get_targets(pre, ['all'])
        searchStr = what[1].lower()
        custom_cols = [col for col in tmp.columns if searchStr in col.lower()]

    # handle custom target input
    if what[0] == 'custom_id':
        print('custom_id')
        tmp = get_targets(pre, ['all'])
        what[0] = 'ID'
        target_tmp = tmp[what]

    return target_tmp

# transform SNPs to numbers
def recode_snps(snp_frame, snp_names):
    print('\nRecoding SNPs...')
    # deep copy snp_frame
    snp_frame_recode = snp_frame.copy()
    # transform snp data to numbers
    for snpID in snp_frame[snp_names]:  # only recode SNPs (not ID :-)
        print('Recoding SNP ' + snpID + ' ...')
        ascii_sum = []
        for x in snp_frame[snpID]:
            if (x[0] != x[1]):  # hetero --> 0
                ascii_sum.append(1)
            else:   #--> else: ascii sum over both characters
                ascii_sum.append(ord(x[0]) + ord(x[1]))

        # set highest ascii-sum to 2, second highest value to 0
        for index in [i for i, x in enumerate(ascii_sum) if x == np.max(ascii_sum)]:
            ascii_sum[index] = 3
        for index in [i for i, x in enumerate(ascii_sum) if x == np.max(ascii_sum)]:
            ascii_sum[index] = 2

        # recode rare alleles?

        # write recoded values to snp_frame
        snp_frame_recode[snpID] = ascii_sum

    #.get_dummies(s)
    return snp_frame_recode

# one hot encode snp matrix
def one_hot_snps(snp_frame, snp_names):
    print('\nOne-hot-encoding SNPs...')
    snp_frame_oneHot = pandas.DataFrame()
    snp_frame_oneHot['ID'] = snp_frame['ID']    # add ID col
    for snpID in snp_names:
        single = pandas.get_dummies(snp_frame[snpID])

        if single.shape[1] == 1:    # handle snps with no variance
            snp_frame_oneHot[snpID + '__' + str(1)] = single[2]
        else:
            for i in range(single.shape[1]):    # assign new snp names and get one-hot encoded snps
                snp_frame_oneHot[snpID + '__' + str(i+1)] = single[i+1]

    # get new one-hot snp names
    snp_names_oneHot = snp_frame_oneHot.columns[1:].values
    return snp_frame_oneHot, snp_names_oneHot

# get feature importance
def get_feature_importance(results, feature_names, data, targets, roiName):
    print('\nComputing Feature Importance...')
    results_tree = results.result_tree
    best_config = results_tree.get_best_config_for(outer_cv_fold=0)
    imp_tmp = pandas.DataFrame(index=[np.arange(1, len(best_config.best_config_object_for_validation_set.fold_list)+1)], columns=feature_names)
    i = 1
    for inner_fold in best_config.best_config_object_for_validation_set.fold_list:
        f_imp = inner_fold.test.feature_importances_
        if len(f_imp) > 0:
            t = results.inverse_transform_pipeline(best_config.config_dict, data, targets, f_imp)
            try:
                imp_tmp.loc[i, :] = t
            except ValueError:
                print('opps t')

        else:
            imp_tmp.loc[i, :] = []
        i += 1
    importance_scores_mean = pandas.DataFrame(index=[roiName], columns=feature_names)
    importance_scores_std = pandas.DataFrame(index=[roiName], columns=feature_names)
    for snpID in feature_names:
        importance_scores_mean.loc[roiName, snpID] = np.median(imp_tmp[snpID])
        importance_scores_std.loc[roiName, snpID] = np.std(imp_tmp[snpID])
        #importance_scores.loc[roiName, snpID] = imp_tmp[snpID].tolist()
    return importance_scores_mean, importance_scores_std

# setup photon HP
def setup_model():
    from Framework.PhotonBase import PipelineElement, PipelineSwitch, Hyperpipe
    from sklearn.model_selection import KFold

    metrics = ['variance_explained', 'pearson_correlation', 'mean_absolute_error']
    my_pipe = Hyperpipe('primary_pipe', optimizer='grid_search',
                        optimizer_params={},
                        best_config_metric='mean_absolute_error',
                        metrics=metrics,
                        inner_cv=KFold(n_splits=20, shuffle=True, random_state=3),
                        eval_final_performance=False,
                        verbose=0)

    # setup hyperpipe

    # # get interaction terms
    # # register elements
    # from Framework.Register import RegisterPipelineElement
    # photon_package = 'PhotonCore'  # where to add the element
    # photon_name = 'interaction_terms'  # element name
    # class_str = 'sklearn.preprocessing.PolynomialFeatures'  # element info
    # element_type = 'Transformer'  # element type
    # RegisterPipelineElement(photon_name=photon_name,
    #                         photon_package=photon_package,
    #                         class_str=class_str,
    #                         element_type=element_type).add()

    # # add the elements
    # my_pipe += PipelineElement.create('interaction_terms', {'degree': [2]},  interaction_only=True,
    #                                   include_bias=False, test_disabled=False)


    # remove 0-variance features
    my_pipe += PipelineElement.create('ZeroVarianceFilter')

    # add feature selection
    my_pipe += PipelineElement.create('CategoricalANOVASelectPercentile', {'percentile': [5]}, test_disabled=False)

    #tree_estimator = PipelineElement.create('RandomForestRegressor', {'min_samples_split': [10, 30, 80, 100]}, n_estimators=100)
    svr_estimator = PipelineElement.create('SVR', {'kernel': ['linear'], 'C': [1.0]})
    #svr_estimator = PipelineElement.create('SVR', {'kernel': ['linear', 'rbf'], 'C': [.001, .01, 0.7, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 100.0]})
    #KNNreg = PipelineElement.create('KNeighborsRegressor')
    #my_pipe += PipelineSwitch('final_estimator', [svr_estimator, tree_estimator, KNNreg])

    #my_pipe += tree_estimator
    my_pipe += svr_estimator
    return my_pipe, metrics

###############################################################################################################
# iterate over targets (e.g. ROI volumes)
def run_analysis(data_dict):

    data = data_dict['data']
    targets = data_dict['targets']
    roiName = data_dict['roiName']
    snp_names = data_dict['snpNames']
    remove_covs = data_dict['remove_covs']
    getImportanceScores = data_dict['getImportanceScores']
    perm_test = data_dict['perm_test']

    # create PHOTON hyperpipe
    my_pipe, metrics = setup_model()

    # shuffle targets if running a permutation test
    if perm_test == True:
        print('\nPERMUTATION TEST: SHUFFLING TARGETS NOW!')
        np.random.shuffle(targets)

    # remove confounders from target data (age, gender, site, ICV)
    if remove_covs:
        print('\nRemoving covariates from targets.')
        import statsmodels.api as sm
        # # convert all covs to numeric
        # for c in data_dict['covs']:
        #     data_dict[c] = pandas.to_numeric(data_dict[c])
        ols_X = data_dict['covs']
        ols_X = sm.add_constant(ols_X)
        ols_model = sm.OLS(targets, ols_X)
        ols_results = ols_model.fit()
        targets = np.asarray(ols_results.resid)

    # fit PHOTON model
    results = my_pipe.fit(data, targets)
    results_tree = results.result_tree

    # get feature importance
    if getImportanceScores:
        importance_scores = get_feature_importance(results=results, feature_names=snp_names, data=data, targets=targets, roiName=roiName)
    else:
        importance_scores = []

    # TEST SET -> Test
    #best_config_performance_test = results_tree.get_best_config_performance_validation_set(outer_cv_fold=1)     # when outer fold is active
    best_config_performance_test = results_tree.get_best_config_performance_validation_set()  # when outer fold is in-active

    # TEST SET -> Train
    best_config_performance_train = results_tree.get_best_config_performance_validation_set(train_data=True)

    print('\n\nBest config performance TEST: ' + roiName + ' ' + str(best_config_performance_test))
    print('Best config performance TRAIN: ' + roiName + ' ' + str(best_config_performance_train))

    # initialize results DataFrame
    mets_train = pandas.DataFrame(index=[roiName], columns=metrics)
    mets_test = pandas.DataFrame(index=[roiName], columns=metrics)
    for met in metrics:
        mets_train.loc[roiName, met] = np.median(best_config_performance_train[met])
        mets_test.loc[roiName, met] = np.median(best_config_performance_test[met])

    return mets_test, mets_train, importance_scores, metrics
