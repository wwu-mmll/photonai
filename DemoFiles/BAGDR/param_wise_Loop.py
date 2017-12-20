from helpers import get_data, run_analysis
import numpy as np
import pandas


# ---------------------------
# run the analysis
if __name__ == '__main__':
    ###############################################################################################################
    # pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    pre = '/spm-data/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
    # pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/'

    group_id = 2  # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    one_hot_it = False
    target_modalities = ['all']

    remove_covs = True
    covs_out = ['Alter', 'Geschlecht', 'Site']

    discretize_targets = True

    getImportanceScores = False
    perm_test = True
    n_perms = 3

    ###############################################################################################################
    # get data
    df, ROI_names, snp_names = get_data(pre=pre, one_hot_it=one_hot_it, what=target_modalities)

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    # group_id = 2
    df = df.loc[df['Group'] == group_id]
    print(str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

    import time
    millis1 = int(round(time.time()))
    # Execute parallel
    import multiprocessing as mp

    for permInd in range(n_perms):
        print('Running permutation ' + str(permInd) + '/' + str(n_perms))

        # PREPARE DATA AND TARGETS to be passed to the parallel function
        data_dict_list = []
        for roiName in ROI_names:
            print('\n\n\n\n' + roiName + '...')

            # Filter out NaN targets
            df_tmp = df.copy()  # deep copy the dataframe so we can drop samples without loosing them in the next iteration
            df_tmp = df_tmp.dropna(subset=[roiName])  # Filter those samples whose current targets are NaN
            print(str(df_tmp.shape[0]) + '/' + str(df.shape[0]) + ' samples remaining.')

            # get targets
            roi_targets = np.asarray(df_tmp[roiName])

            # discretize targets
            print('\nRounding targets.\n')
            if discretize_targets:
                roi_targets = np.around(roi_targets, decimals=1)

            # get data (numeric snps)
            roi_data = np.asarray(df_tmp[snp_names])
            roi_data = roi_data.copy(order='C')  # fixes an error (don't know why this is necessary)

            # get covs
            covs = df_tmp[covs_out ]

            # move all relevant info a dict so it can be passed to the mpPool
            data_dict_list.append({'data': roi_data, 'targets': roi_targets, 'roiName': roiName, 'snpNames': snp_names,
                                   'covs': covs, 'remove_covs': remove_covs, 'getImportanceScores': getImportanceScores,
                                   'perm_test': perm_test})

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
            if getImportanceScores:
                importance_scores_summary = importance_scores_summary.append(imp)

            # save metrics summary
            if perm_test:
                metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test_perm_' + str(permInd))
                metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train_perm_' + str(permInd))
            else:
                # metrics_summary_test = metrics_summary_test.sort_values(by='variance_explained', axis=0, ascending=False)
                # metrics_summary_train = metrics_summary_train.sort_values(by='variance_explained', axis=0, ascending=False)
                metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test')
                metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train')

            if getImportanceScores and ~perm_test:
                importance_scores_summary.to_pickle(path=pre + 'Results/importance_scores_summary')

        # test = pandas.read_pickle(path=pre + 'Results//metrics_summary_test')

    p_list = perm_test(n_perms=n_perms, alpha=.05, pre=pre)

print('')

