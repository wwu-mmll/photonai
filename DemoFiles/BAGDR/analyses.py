from helpers import get_data, run_analysis
import numpy as np
import pandas
from Logging.Logger import Logger

# ---------------------------
# run the analysis
if __name__ == '__main__':
    metrics = ['variance_explained', 'pearson_correlation', 'mean_absolute_error']

    #pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    #pre = '/spm-data/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
    pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    results_file = pre + 'Results/metrics_summary'
    importance_scores_file = pre + 'Results/importance_scores'

    n_perms = 47  # 0 = no permutation
    getImportanceScores = True
    #----------------------------------------------------------------------------------------


    if n_perms > 0:
        perm_test_bool = True
    else:
        perm_test_bool = False

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


    # 1. get results table for explained variance per ROI
    from investigate_results import perm_hist, get_results_tab, star_plot, get_imps

    metric = metrics[0]
    results_table = get_results_tab(alpha_cor=.05, metrics_summary_test=results_file + '_test',
                                    p_cor_file=results_file + '_p_cor', metric=metric)

    # plot significance level and null-distribution for all sig. ROIs
    h = perm_hist(metrics_summary_test=results_file + '_test',
                  p_cor_file=results_file + '_p_cor',
                  perm_vec_file=results_file + '_permVec',
                  metric=metric,
                  figure_file=results_file + '_perm_plot.png')

    # 2. Analyses using model fit
    # a. model fit is higher in bad outcome(high BDI) patients
    # b. r(model fit, CTQ) < 0
    # c. predict environmental risk group using model fit
    # get true and predicted target per person and ROI


    # 3. - Insight re ROIs
    # - ANOVA: Different snp importance for vol, surf, thick?
    # - PCA, t-SNE visualization; color code vol, surf, thick
    # ? cluster ROIs based on (sig) SNPs? -> Clusters useful? e.g. different for vol, surf...

    # get SNP importance
    imps_all = get_imps(importance_scores_file=importance_scores_file, alpha_cor=.05, sig_inds=results_table.index.values)
    import time

    # # Star Plots of SNP importance (per ROI)
    # cat = list(imps_all.columns.values)   # get SNP names
    # for roi in imps_all.index.values:
    #     imps = np.abs(imps_all.loc[roi, :].values)
    #     imps = ((imps - imps.min()) / (imps.max() - imps.min()))  # scale imp scores 0 to 100
    #     imps = list(imps)
    #     star_plot(title=roi, cat=cat, values=imps)
    #     time.sleep(5)

    # # 4. insights re SNPs
    # # Star Plots for SNP importance (per SNP)
    # cat = list(imps_all.index.values)   # get SNP names
    # for snp in imps_all.columns.values:
    #     imps = np.abs(imps_all.loc[:, snp].values)
    #     imps = ((imps - imps.min()) / (imps.max() - imps.min()))  # scale imp scores 0 to 100
    #     imps = list(imps)
    #     star_plot(title=snp, cat=cat, values=imps)
    #     time.sleep(5)

    # 5.
    # Utility checks
    # - Use sig ROI parameters to predict MDD vs. HC

    # - prediction of states (BDI) not possible?


    # - Use sig SNPs to predict MDD vs. HC

    # - prediction of states (BDI) not possible?

    # Outlook
    # exp_variance vs. n_train graph (for Lianne)

    Logger.log('Done')
