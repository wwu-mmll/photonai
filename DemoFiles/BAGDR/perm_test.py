import pandas
import numpy as np

def perm_test_importance(n_perms, importance_score_file):
    # get true vals (no permutation)
    true_imps = pandas.read_pickle(path=importance_score_file)
    p_imp = true_imps.copy()

    return p_imp

    # get permutations
def perm_test_multCompCor(n_perms, results_file, metrics):
    # get true vals (no permutation)
    true_vals = pandas.read_pickle(path=results_file + '_test')
    p_vals = true_vals.copy()

    # get permutations
    for metric in metrics:
        perms = []
        for permInd in range(1, n_perms+1):
            a = pandas.read_pickle(path=results_file + '_test_perm_' + str(permInd))[metric]
            perms.append(np.asarray(a.tolist()))

        # put all p_values under permutation in one vector
        perm_vec = np.asarray([item for sublist in perms for item in sublist])
        perm_vec = np.nan_to_num(perm_vec)

        # get p-values
        met_true = true_vals[metric]
        for i in p_vals.index:
            p = np.sum(met_true[i] <= perm_vec) / len(perm_vec)
            if p==0:
                p = 1 / len(perm_vec)
            p_vals.loc[i, metric] = p

    return p_vals, perm_vec

