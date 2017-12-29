import pandas
import numpy as np

def perm_test_importance(n_perms, importance_score_file):
    # get true vals (no permutation)
    true_imps_raw = pandas.read_pickle(path=importance_score_file + '_median')
    true_imps_raw = true_imps_raw.transpose()

    # get perm vals and p-values for two-tailed, ROI-wise CORRECTED test (importance p-values are corrected for all SNPs in the ROI; not across ROIs)
    imp_p_cor = pandas.DataFrame(index=true_imps_raw.index.values, columns=true_imps_raw.columns.values)
    mdf = pandas.DataFrame()
    for permInd in range(1, n_perms + 1):
        a = pandas.read_pickle(path=importance_score_file + '_median_perm_' + str(permInd))
        a = a.transpose()
        mdf = pandas.concat([mdf, a])

    for col_id in imp_p_cor.columns.values:
        perms = mdf.loc[:][col_id].tolist()
        for row_id in imp_p_cor.index.values:
            # get true imp score
            true_imp_scalar = true_imps_raw.loc[row_id][col_id]
            # get p-values for two-tailed, ROI-wise CORRECTED test
            p = np.sum(np.abs(true_imp_scalar) <= np.abs(perms)) / len(perms)
            if p == 0:
                p = 1 / len(perms)
            imp_p_cor.loc[row_id][col_id] = p
    return imp_p_cor

def perm_test_importance_old(n_perms, importance_score_file):
    # get true vals (no permutation)
    true_imps_raw = pandas.read_pickle(path=importance_score_file + '_median')
    true_imps = pandas.DataFrame(index=true_imps_raw.index.values, columns=true_imps_raw.columns.values)
    # get mean over folds
    for col_id in true_imps_raw.columns.values:
        for row_id in true_imps_raw.index.values:
            true_imps.loc[row_id][col_id] = np.mean(true_imps_raw.loc[row_id][col_id])

    # # get perm vals and p-values for two-tailed, uncorrected test
    # imp_p = pandas.DataFrame(index=true_imps.index.values, columns=true_imps.columns.values)
    # for row_id in imp_p.index.values:
    #     for col_id in imp_p.columns.values:
    #         perms = list()
    #         for permInd in range(1, n_perms + 1):
    #             # get mean over folds for all permutations
    #             a = pandas.read_pickle(path=importance_score_file + '_perm_' + str(permInd))
    #             perms.append(np.mean(a.loc[row_id][col_id]))
    #         # get p-values for two-tailed, uncorrected test
    #         p = np.sum(np.abs(true_imps.loc[row_id][col_id]) <= np.abs(perms)) / len(perms)
    #         if p == 0:
    #             p = 1 / len(perms)
    #         imp_p.loc[row_id][col_id] = p

    # get perm vals and p-values for two-tailed, ROI-wise CORRECTED test
    imp_p_cor = pandas.DataFrame(index=true_imps.index.values, columns=true_imps.columns.values)
    for row_id in imp_p_cor.index.values:
        perms = list()
        for col_id in imp_p_cor.columns.values:
            for permInd in range(1, n_perms + 1):
                # get mean over folds for all permutations
                a = pandas.read_pickle(path=importance_score_file + '_perm_' + str(permInd))
                perms.append(np.mean(a.loc[row_id][col_id]))
        for col_id in imp_p_cor.columns.values:
            # get p-values for two-tailed, ROI-wise CORRECTED test
            p = np.sum(np.abs(true_imps.loc[row_id][col_id]) <= np.abs(perms)) / len(perms)
            if p == 0:
                p = 1 / len(perms)
            imp_p_cor.loc[row_id][col_id] = p

    # return imp_p, imp_p_cor
    return imp_p_cor


    # get permutations
def perm_test_multCompCor(n_perms, results_file, metrics):
    # get true vals (no permutation)
    true_vals = pandas.read_pickle(path=results_file + '_test')
    p_vals = true_vals.copy()

    # get permutations
    perm_vec_mets = pandas.DataFrame(columns=metrics)
    for metric in metrics:
        perms = []
        for permInd in range(1, n_perms+1):
            a = pandas.read_pickle(path=results_file + '_test_perm_' + str(permInd))[metric]
            perms.append(np.asarray(a.tolist()))

        # put all p_values under permutation in one vector and store it in a df
        perm_vec = np.asarray([item for sublist in perms for item in sublist])
        perm_vec = np.nan_to_num(perm_vec)
        perm_vec_mets[metric] = perm_vec

        # get p-values for two-tailed test
        met_true = true_vals[metric]
        for i in p_vals.index:
            p = np.sum(np.abs(met_true[i]) <= np.abs(perm_vec)) / len(perm_vec)
            if p==0:
                p = 1 / len(perm_vec)
            p_vals.loc[i, metric] = p

    return p_vals, perm_vec_mets

