import numpy as np
import pandas

def perm_hist(metrics_summary_test, p_cor_file, perm_vec_file, metric, figure_file):
    alpha = .05

    import matplotlib.pyplot as plt
    # get the real performance parameter and p-values
    mets = pandas.read_pickle(path=metrics_summary_test)[metric]
    p_cor = pandas.read_pickle(path=p_cor_file)[metric]

    # get the null distribution (under permutation)
    perm_vec = pandas.read_pickle(path=perm_vec_file)[metric]


    # histogram
    plt.hist(perm_vec)
    #n, bins, patches = plt.hist(perm_vec, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel(metric)
    plt.ylabel('Probability')
    plt.title('Permutation Test (k=' + str(len(perm_vec)) + ' permutations)')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(False)

    # add significant parameters as vertical lines
    sigs = mets[p_cor < alpha]
    #sigs_labels = sigs.index
    plt.vlines(sigs.tolist(), 0, 1, lw=3.0, colors='r')

    o = 3
    for ind in sigs.index:
        print(ind)
        print(sigs[ind])
        print(p_cor[ind])
        plt.vlines(sigs[ind], 0, o, lw=3.0, colors='r')
        plt.text(sigs[ind]+(sigs[ind]*.05), o, (ind + ' (p=' + str(np.around(p_cor[ind], decimals=3)) + ')'), verticalalignment='center')
        o += .25

    plt.savefig(figure_file)#, bbox_inches='tight')
    plt.show()
    print('')
