import pandas
import numpy as np

def perm_test(n_perms, alpha, pre)
    # get permutations
    perms = []
    for permInd in range(n_perms):
        a = pandas.read_pickle(path=pre + 'Results/metrics_summary_test_perm_' + str(permInd))['variance_explained']
        perms.append(np.asarray(a.tolist()))

    flat_list = np.asarray([item for sublist in perms for item in sublist])
    flat_list = np.nan_to_num(flat_list)

    # get results for true targets
    met_true = pandas.read_pickle(path=pre + 'Results/metrics_summary_test')['variance_explained']

    # mark significant
    #sig_thresh = np.percentile(flat_list, 100-(alpha*100))
    #met_true[met_true <= sig_thresh] = 0.0

    # get actual p-values
    met_p = pandas.read_pickle(path=pre + 'Results/metrics_summary_test')['variance_explained']
    for m in met_p:
        p = np.sum(m <= flat_list) / len(flat_list)
        print(str(m) + '; p=' + str(p))

    return p_list
    print('Done')

