# PHOTON Analysis for the Multimodal Brain Atlas of Genetic Depression Risk
import pandas
import numpy as np

#pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
pre = '/spm-data/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
#pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'

global getImp, perm_test
getImp = False

perm_test = False
n_perms = 1

covs_out = True
one_hot_it = False
discretize_targets = True

def get_data():
    # get data (snps; recode SNPs to numerical values; 0-->hetero)
    snp_file = pre + 'Genetics/FORMarburgh_1KG_imputed_MDD_Compound_Genotypes_withSNPids_cleared.xlsx'
    snps_tmp, snp_names = get_snps(file=snp_file)
    snp_num_frame = recode_snps(snps_tmp, snp_names)   # recode SNP strings to numerical values
    if one_hot_it:
        snp_num_frame, snp_names = one_hot_snps(snp_num_frame, snp_names)    # one-hot encode snp matrix

    # get covariates (e.g. diagnosis)
    cov_file = pre + 'Datenbank_Update_DataFreeze1&2_17-11-2017_relVars.csv'
    covs_tmp = get_covs(file=cov_file)

    ##############################################################################################################
    # get targets (e.g. volumes, thickness, ... ); (CAT12_GM, DTI, rs-fMRI_Hubness, ...)
    # Thickness
    target_file = pre + 'FreeSurfer_ROI/CorticalMeasuresENIGMA_ThickAvg.csv'
    target_thick = get_targets(file=target_file)

    # SurfaceArea
    target_file = pre + 'FreeSurfer_ROI/CorticalMeasuresENIGMA_SurfAvg.csv'
    target_surf = get_targets(file=target_file)

    # Volume
    target_file = pre + 'FreeSurfer_ROI/LandRvolumes.csv'
    target_vol = get_targets(file=target_file)

    # CAT12 Vgm
    target_file = pre + 'CAT12_ROI/ROI_CAT12_r1184_catROI_neuromorphometrics_Vgm.csv'
    target_Vgm = get_targets(file=target_file)

    # merge target datasets and drop duplicate cols (e.g. ICV)
    target_tmp = pandas.merge(target_thick, target_surf, how='inner', on='ID')
    target_tmp = pandas.merge(target_tmp, target_vol, how='inner', on='ID')
    target_tmp = pandas.merge(target_tmp, target_Vgm, how='inner', on='ID')

    # ToDo: drop duplicate cols

    ROI_names = list(target_tmp.columns[1:].values)  # skip subID and take the rest
    ###############################################################################################################

    # merge the three dataframes into one dataframe and only keep the intersection (via 'inner')
    df = pandas.merge(covs_tmp, target_tmp, how='inner', on='ID')
    df = pandas.merge(df, snp_num_frame, how='inner', on='ID')
    print('Target shape: ' + str(target_tmp.shape))
    print('Covs shape: ' + str(covs_tmp.shape))
    print('SNPs shape: ' + str(snps_tmp.shape))
    print('Intersection merge: ' + str(df.shape))
    return df, ROI_names, snp_names

# get Genetic data
def get_snps(file):
    snp_frame = pandas.read_excel(file)                    # read snp data
    snp_frame = snp_frame.dropna(axis=0, how='any')     # get rid of subjects whose data contains NaNs
    snp_names = snp_frame.columns[1:].values            # skip subID and take the rest
    snp_frame['ID'] = snp_frame['ID'].astype(str)       # cast ID to string
    return snp_frame, snp_names

# get clinical/psychometric etc. variables
def get_covs(file):
    covs_frame = pandas.read_csv(file)                 # read cov data
    covs_frame.columns = ['ID' if x == 'Proband' else x for x in covs_frame.columns]    # rename the ID column to allow merge
    covs_frame = covs_frame.dropna(axis=0, how='any', subset=['Group'])         # get rid of subjects whose data contains NaNs in the listed col(s) (we only need Group to be valid for everyone)
    covs_frame['ID'] = covs_frame['ID'].astype(str)
    subjID_covs= [s[0:4].lstrip("0") for s in covs_frame['ID']]  # only the first 5 characters are relevant, also remove leading zeros
    covs_frame = covs_frame.assign(ID=subjID_covs)
    return covs_frame

# get targets (ROI-wise cortical thickness or volumes or ...)
def get_targets(file):
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

# transform SNPs to numbers
def recode_snps(snp_frame, snp_names):
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
    results_tree = results.result_tree
    best_config = results_tree.get_best_config_for(outer_cv_fold=0)
    imp_tmp = pandas.DataFrame(index=[np.arange(1, len(best_config.best_config_object_for_validation_set.fold_list)+1)], columns=feature_names)
    i=1
    for inner_fold in best_config.best_config_object_for_validation_set.fold_list:
        f_imp = inner_fold.test.feature_importances_
        if len(f_imp) > 0:
            t = results.inverse_transform_pipeline(best_config.config_dict, data, targets, f_imp)
            imp_tmp.loc[i, :] = t[0]

        else:
            imp_tmp.loc[i, :] = []
        i += 1
    importance_scores = pandas.DataFrame(index=[roiName], columns=feature_names)
    for snpID in feature_names:
        importance_scores.loc[roiName, snpID] = imp_tmp[snpID].tolist()
    return importance_scores

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


# iterate over targets (e.g. ROI volumes)
def run_analysis(data_dict):

    data = data_dict['data']
    targets = data_dict['targets']
    roiName = data_dict['roiName']
    snp_names = data_dict['snpNames']
    covs = data_dict['covs']

    # create PHOTON hyperpipe
    my_pipe, metrics = setup_model()

    # shuffle targets if running a permutation test
    if perm_test == True:
        print('PERMUTATION TEST: SHUFFLING TARGETS NOW!')
        np.random.shuffle(targets)

    # remove confounders from target data (age, gender, site, ICV)
    if covs_out:
        import statsmodels.api as sm
        ols_X = covs
        ols_X = sm.add_constant(ols_X)
        ols_model = sm.OLS(targets, ols_X)
        ols_results = ols_model.fit()
        targets = np.asarray(ols_results.resid)
        print('Removing covariates from targets.')

    # fit PHOTON model
    results = my_pipe.fit(data, targets)
    results_tree = results.result_tree

    # get feature importance
    if getImp:
        importance_scores = get_feature_importance(results=results, feature_names=snp_names, data=data, targets=targets, roiName=roiName)
    else:
        importance_scores = []

    # TEST SET -> Test
    #best_config_performance_test = results_tree.get_best_config_performance_validation_set(outer_cv_fold=1)     # when outer fold is active
    best_config_performance_test = results_tree.get_best_config_performance_validation_set()  # when outer fold is in-active

    # TEST SET -> Train
    best_config_performance_train = results_tree.get_best_config_performance_validation_set(train_data=True)


    print('Best config performance TEST: ' + roiName + ' ' + str(best_config_performance_test))
    print('Best config performance TRAIN: ' + roiName + ' ' + str(best_config_performance_train))

    # initialize results DataFrame
    mets_train = pandas.DataFrame(index=[roiName], columns=metrics)
    mets_test = pandas.DataFrame(index=[roiName], columns=metrics)
    for met in metrics:
        mets_train.loc[roiName, met] = np.median(best_config_performance_train[met])
        mets_test.loc[roiName, met] = np.median(best_config_performance_test[met])

    return mets_test, mets_train, importance_scores


# ---------------------------
# run the analysis
if __name__ == '__main__':
    # get data
    global df
    df, ROI_names, snp_names = get_data()

    # add a few covs to be tested as well
    # ROI_names.append('BMI')
    # ROI_names.append('BDI_Sum')
    # ROI_names.append('CTQ_Sum')
    # ROI_names.append('IQ')

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    group_id = 2
    df = df.loc[df['Group'] == group_id]
    print(str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

    import time
    millis1 = int(round(time.time()))
    # Execute parallel
    import multiprocessing as mp

    for permInd in range(n_perms):
        print('Running permutation ' + str(permInd) + '/' + str(n_perms))
        # PREPARE DATA AND TARGETS
        data_dict_list = []

        for roiName in ROI_names:
            print('\n\n\n\n' + roiName + '...')
            # print(pandas.__version__)

            # Filter samples
            df_tmp = df.copy()  # deep copy the dataframe so we can drop samples without loosing them in the next iteration
            df_tmp = df_tmp.dropna(subset=[roiName])  # Filter those samples whose current targets are NaN
            print(str(df_tmp.shape[0]) + '/' + str(df.shape[0]) + ' samples remaining.')

            # get targets
            roi_targets = np.asarray(df_tmp[roiName])

            # get data (numeric snps)
            roi_data = np.asarray(df_tmp[snp_names])
            roi_data = roi_data.copy(order='C')  # fixes an error (don't know why this is necessary)

            # discretize targets
            print('\nRounding targets.\n')
            if discretize_targets:
                roi_targets = np.around(roi_targets, decimals=1)

            # get covs
            covs = df_tmp[['Alter', 'Geschlecht', 'Site']]
            data_dict_list.append({'data': roi_data, 'targets': roi_targets, 'roiName': roiName, 'snpNames': snp_names, 'covs': covs})

        results = mp.Pool().map(run_analysis, data_dict_list)
        millis2 = int(round(time.time()))
        print('Time (minutes): ' + str((millis2 - millis1) / 60))

        # initialize results DataFrame and add/join results from parallel pool
        metrics_summary_train = pandas.DataFrame()
        metrics_summary_test = pandas.DataFrame()
        importance_scores_summary = pandas.DataFrame()
        for roi in results:
            te, tr, imp = roi
            metrics_summary_test = metrics_summary_test.append(te)
            metrics_summary_train = metrics_summary_train.append(tr)
            if getImp:
                importance_scores_summary = importance_scores_summary.append(imp)

        # save metrics summary
        if perm_test:
            metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test_perm_' + str(permInd))
            metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train_perm_' + str(permInd))
        else:
            metrics_summary_test = metrics_summary_test.sort_values(by='variance_explained', axis=0, ascending=False)
            metrics_summary_train = metrics_summary_train.sort_values(by='variance_explained', axis=0, ascending=False)
            metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test')
            metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train')

        if getImp and ~perm_test:
            importance_scores_summary.to_pickle(path=pre + 'Results/importance_scores_summary')

        print('Done')
    # test = pandas.read_pickle(path=pre + 'Results//metrics_summary_test')
    print('Done')

