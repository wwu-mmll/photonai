import pandas
import numpy as np

#pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
#pre = '/spm-data/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
#pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
pre = 'T:/spielwiese_tim\BrainAtlasOfGeneticDepressionRisk/'

# get permutations
alpha = .05   # percent alpha
n_perms = 10
perms = []
for permInd in range(n_perms):
    a = pandas.read_pickle(path=pre + 'Results/metrics_summary_test_perm_' + str(permInd))['variance_explained']
    perms.append(np.asarray(a.tolist()))

flat_list = np.asarray([item for sublist in perms for item in sublist])
flat_list = np.nan_to_num(flat_list)
#max = np.max(flat_list)
sig_thresh = np.percentile(flat_list, 100-(alpha*100))


# get result for true targets
met_true = pandas.read_pickle(path=pre + 'Results/metrics_summary_test')['variance_explained']
met_true[met_true < sig_thresh] = 0.0
print('Done')

