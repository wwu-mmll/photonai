import numpy as np
import pandas
from Logging.Logger import Logger


def get_results(metrics_summary_test, p_cor_file, metric):
    # get the real performance parameter and p-values
    mets = pandas.read_pickle(path=metrics_summary_test)[metric].to_frame()
    p_cor = pandas.read_pickle(path=p_cor_file)[metric].to_frame()
    p_cor = p_cor.rename(columns={metric: "p_cor"})
    return mets, p_cor

def get_results_tab(metrics_summary_test, p_cor_file, metric, alpha_cor = .05):
    mets, p_cor = get_results(metrics_summary_test=metrics_summary_test, p_cor_file=p_cor_file, metric=metric)
    res_tab = mets.join(p_cor)
    res_tab = res_tab.loc[res_tab['p_cor'] < alpha_cor]
    res_tab = res_tab.sort_values(by='variance_explained', axis=0, ascending=False)
    with pandas.option_context('display.max_rows', None, 'display.max_columns', 3):
        Logger().info(res_tab)
    return res_tab

def get_imps(importance_scores_file, alpha_cor = .05, sig_inds = []):
    imps_scores = pandas.read_pickle(path=importance_scores_file + '_median')
    #imps_std_all = pandas.read_pickle(path=importance_scores_file + '_std')
    imps_p_all = pandas.read_pickle(path=importance_scores_file + '_p_corrected').transpose()

    if sig_inds != []:  # show only sig rois?
        imps_scores = imps_scores.loc[sig_inds, :]
        imps_p_all = imps_p_all.loc[sig_inds, :]

    imps_tab = imps_scores.mask(imps_p_all >= alpha_cor, 0)
    imps_tab = imps_tab.loc[:, (imps_tab != 0).any(axis=0)]
    return imps_tab

def perm_hist(metrics_summary_test, p_cor_file, perm_vec_file, metric, figure_file, alpha_cor = .05):
    # p-vals are already correct for multiple comparisons so this will do
    import matplotlib.pyplot as plt
    # get the real performance parameter and p-values
    res_tab = get_results_tab(alpha_cor=alpha_cor, metrics_summary_test=metrics_summary_test, p_cor_file=p_cor_file, metric=metric)
    res_tab = res_tab.sort_values(by='variance_explained', axis=0, ascending=True)  # sort ascending for plot
    mets = res_tab.loc[:, metric]
    p_cor = res_tab.loc[:, 'p_cor']

    # get the null distribution (under permutation)
    perm_vec = pandas.read_pickle(path=perm_vec_file)[metric]

    # histogram
    bin_n, _, _ = plt.hist(perm_vec, bins=100)
    #n, bins, patches = fig.hist(perm_vec, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.title('Permutation Test (k=' + str(len(perm_vec)) + ' permutations)')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add significant parameters as vertical lines
    sigs = mets
    #sigs_labels = sigs.index
    o = np.max(bin_n)
    for ind in sigs.index:
        plt.vlines(sigs[ind], 0, o, lw=3.0, colors='r')
        plt.text(sigs[ind]+(sigs[ind]*.05), o, (ind + ' (p=' + str(np.around(p_cor[ind], decimals=3)) + ')'), verticalalignment='center')
        o -= np.max(bin_n)/20

    plt.subplots_adjust(right=.55)
    plt.savefig(figure_file)#, bbox_inches='tight')
    plt.show()

def star_plot(title, cat, values):
    from math import pi
    import matplotlib.pyplot as plt

    values = [i * 100 for i in values]  # multiply each element by 100

    N = len(cat)

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # Because our chart will be circular we need to append a copy of the first
    # value of each list at the end of each list with data
    values += values[:1]
    x_as += x_as[:1]

    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Create polar plot
    ax = plt.subplot(111, polar=True)

    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set position of y-labels
    ax.set_rlabel_position(1)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.25)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.25)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    plt.yticks([20, 40, 60, 80, 100], [".2", ".4", ".6", ".8", "1"])

    # Plot data
    a = zip(x_as, values)
    for x, y in a:
        ax.plot((0, x), (0, y), linewidth=3, linestyle='solid', color='blue', zorder=3)
    #ax.scatter(x_as, values, linewidth=3, linestyle='solid', zorder=3)

    # Fill area
    #ax.fill(x_as, values, 'b', alpha=0.3)

    # Set axes limits
    plt.ylim(0, 100)

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi

        if angle_rad == 0:
            ha, distance_ax = "center", 10
        elif 0 < angle_rad < pi:
            ha, distance_ax = "left", 1
        elif angle_rad == pi:
            ha, distance_ax = "center", 1
        else:
            ha, distance_ax = "right", 1

        ax.text(angle_rad, 100 + distance_ax, cat[i], size=10, horizontalalignment=ha, verticalalignment="center")

    plt.title(title + '\n')
    # Show polar plot
    plt.show()
