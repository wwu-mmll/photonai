from helpers import get_data, run_analysis
import numpy as np
import pandas
from Logging.Logger import Logger


# ---------------------------
# run the analysis
if __name__ == '__main__':
    ###############################################################################################################
    metrics = ['variance_explained', 'pearson_correlation', 'mean_absolute_error']

    #pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    #pre = '/spm-data/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
    pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'

    group_id = 2  # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    one_hot_it = False
    target_modalities = ['all']

    remove_covs = True
    covs_out = ['Alter', 'Geschlecht', 'Site']

    discretize_targets = True

    getImportanceScores = True
    n_perms = 0     # 0 = no permutation

    results_file = pre + 'Results/metrics_summary'
    importance_scores_file = pre + 'Results/importance_scores'

    ###############################################################################################################
    # get data
    df, ROI_names, snp_names = get_data(pre=pre, one_hot_it=one_hot_it, what=target_modalities)

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    # group_id = 2
    df = df.loc[df['Group'] == group_id]
    Logger().info(str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

    import time
    millis1 = int(round(time.time()))
    # Execute parallel
    import multiprocessing as mp
    perm_test_bool = False
    for permInd in range(n_perms+1):
        if permInd == 0:
            Logger().info('Running analysis.')
            perm_test_bool = False
        elif permInd > 0:
            Logger().info('Running permutation ' + str(permInd) + '/' + str(n_perms))
            perm_test_bool = True

        # PREPARE DATA AND TARGETS to be passed to the parallel function
        data_dict_list = []

        #
        # ############
        # # Debug only
        # ROI_names = ROI_names[:4]

        for roiName in ROI_names:
            Logger().info('\n\n\n\n' + roiName + '...')

            # Filter out NaN targets
            df_tmp = df.copy()  # deep copy the dataframe so we can drop samples without loosing them in the next iteration
            df_tmp = df_tmp.dropna(subset=[roiName])  # Filter those samples whose current targets are NaN
            Logger().info(str(df_tmp.shape[0]) + '/' + str(df.shape[0]) + ' samples remaining.')

            # get targets
            roi_targets = np.asarray(df_tmp[roiName])

            # discretize targets
            Logger().info('\nRounding targets.\n')
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
                                   'perm_test': perm_test_bool})

        results = mp.Pool(processes=2).map(run_analysis, data_dict_list)
        millis2 = int(round(time.time()))
        Logger().info('Time (minutes): ' + str((millis2 - millis1) / 60))

        # initialize results DataFrame and add/join results from parallel pool
        metrics_summary_train = pandas.DataFrame()
        metrics_summary_test = pandas.DataFrame()
        importance_scores_summary_median = pandas.DataFrame()
        importance_scores_summary_std = pandas.DataFrame()
        for roi in results:
            te, tr, imp, metrics = roi
            metrics_summary_test = metrics_summary_test.append(te)
            metrics_summary_train = metrics_summary_train.append(tr)
            if getImportanceScores:
                importance_scores_summary_median = importance_scores_summary_median.append(imp[0])
                importance_scores_summary_std = importance_scores_summary_std.append(imp[1])
                #importance_scores_summary['std'] = importance_scores_summary.append(imp)

            metrics_summary_test = metrics_summary_test.sort_values(by='variance_explained', axis=0, ascending=False)
            metrics_summary_train = metrics_summary_train.sort_values(by='variance_explained', axis=0, ascending=False)

            # save metrics summary
            if perm_test_bool:
                metrics_summary_test.to_pickle(path=results_file + '_test_perm_' + str(permInd))
                metrics_summary_train.to_pickle(path=results_file + '_train_perm_' + str(permInd))
                if getImportanceScores:
                    importance_scores_summary_median.to_pickle(path=importance_scores_file + '_median_perm_' + str(permInd))
                    importance_scores_summary_std.to_pickle(path=importance_scores_file + '_std_perm_' + str(permInd))
            else:
                metrics_summary_test.to_pickle(path=results_file + '_test')
                metrics_summary_train.to_pickle(path=results_file + '_train')
                if getImportanceScores:
                    importance_scores_summary_median.to_pickle(path=importance_scores_file + '_median')
                    #importance_scores_summary_std.to_pickle(path=importance_scores_file + '_std')

    # run permutation test
    if perm_test_bool:
        Logger().info('Retrieving Permutation Test info...')
        from perm_test import perm_test_multCompCor
        p_cor, perm_vec_mets = perm_test_multCompCor(n_perms=n_perms, results_file=results_file, metrics=metrics)
        p_cor.to_pickle(path=results_file + '_p_cor')
        perm_vec_mets.to_pickle(path=results_file + '_permVec')

        # get p-vals for importance scores
        if getImportanceScores:
            Logger().info('Retrieving Permutation Test info for Importance Scores...')
            from perm_test import perm_test_importance
            imp_p_cor = perm_test_importance(n_perms=n_perms, importance_score_file=importance_scores_file)
            imp_p_cor.to_pickle(path=importance_scores_file + '_p_corrected')


    # investigate results
    from investigate_results import perm_hist
    metric = metrics[0]
    h = perm_hist(metrics_summary_test=results_file + '_test',
                  p_cor_file=results_file + '_p_cor',
                  perm_vec_file=results_file + '_permVec',
                  metric=metric,
                  figure_file=results_file + 'perm_plot.png')

    Logger().info('')

