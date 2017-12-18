from DemoFiles.MultiTask.helpers import get_data
from DemoFiles.MultiTask.setup_model_MTL import setup_model_MTL
from DemoFiles.MultiTask.setup_model_MTL_2 import setup_model_MTL_2
import numpy as np
import pandas

# ---------------------------
# run the analysis
if __name__ == '__main__':

    ###############################################################################################################
    #pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    pre = '/spm-data-cached/Scratch/spielwiese_tim/BrainAtlasOfGeneticDepressionRisk/'
    #pre = '/home/nils/data/GeneticBrainAtlas/'
    #pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'

    group_id = 2    # 'NaN'=use everyone, 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    #target_modalities = ['custom_str', 'hip']
    target_modalities = ['all']
    #target_modalities = ['custom_id', 'Lhippo', 'Rhippo', 'lHip', 'rHip']

    one_hot_it = False
    discretize_targets = True

    getImportanceScores = False
    perm_test = False

    remove_covs = True
    covs = ['Alter', 'Geschlecht', 'Site', 'TIV']
    ###############################################################################################################

    # get data
    df, ROI_names, snp_names = get_data(pre=pre, one_hot_it=one_hot_it, what=target_modalities,
                                        impute_targets='mean')  # technically, we should always use drop here

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    if not group_id == 2:
        df = df.loc[df['Group'] == group_id]
        print('\n' + str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

    # Prepare multi task targets
    targets = np.asarray(df[ROI_names])

    # create list of dictionaries that define target_info
    target_info = []
    for roi_name in ROI_names:
        output_node_dict = {'name': roi_name, 'target_dimension': 1, 'activation': 'linear',
                            'loss': 'mse', 'loss_weight': 1}
        target_info.append(output_node_dict)

    # shuffle targets if running a permutation test
    if perm_test:
        print('\nPERMUTATION TEST: SHUFFLING TARGETS NOW!')
        np.random.shuffle(targets)

    # remove confounders from target data (e.g. age, gender, site...)
    if remove_covs:
        print('\nRemoving covariates from targets.')
        import statsmodels.api as sm
        # convert all covs to numeric
        for c in covs:
            df[c] = pandas.to_numeric(df[c])
        ols_X = df[covs]
        ols_X = sm.add_constant(ols_X)
        ols_model = sm.OLS(targets, ols_X)
        ols_results = ols_model.fit()
        targets = np.asarray(ols_results.resid)

    # scale targets
    print('\nScaling targets.\n')
    from sklearn.preprocessing import StandardScaler
    targets = StandardScaler().fit_transform(targets)

    # discretize targets
    print('\nRounding targets.\n')
    if discretize_targets:
        targets = np.around(targets, decimals=1)

    # get data (numeric snps)
    data = np.asarray(df[snp_names])
    data = data.copy(order='C')  # fixes an error (don't know why this is necessary)

    # test ANOVA
    from scipy import stats
    snp_in = []
    for snpInd in range(data.shape[1]):
        snpBool = False
        for targetInd in range(targets.shape[1]):
            a = targets[data[:, snpInd] == 1, targetInd]
            b = targets[data[:, snpInd] == 2, targetInd]
            c = targets[data[:, snpInd] == 3, targetInd]

            f, p = stats.f_oneway(a,b,c)

            if p<.001:
                print('One-way ANOVA - snp_name ' + snp_names[snpInd] + '; ' + ROI_names[targetInd])
                print('=============')
                print('F value:', f)
                print('P value:', p, '\n')
                snpBool = True
        if snpBool:
            snp_in.append(snp_names[snpInd])

    data = np.asarray(df[snp_in])
    data = data.copy(order='C')

    # create PHOTON hyperpipe
    my_pipe, metrics = setup_model_MTL_2(target_info=target_info)
    # fit PHOTON model
    import time
    millis1 = int(round(time.time()))
    results = my_pipe.fit(data, targets)
    millis2 = int(round(time.time()))
    print('\nTime elapsed (minutes): ' + str((millis2 - millis1) / 60))
    results_tree = results.result_tree
    best_config_performance_test = results_tree.get_best_config_performance_validation_set()
    best_config_performance_train = results_tree.get_best_config_performance_validation_set(
        train_data=True)
    print('best_config_performance_test:', best_config_performance_test)
    print('best_config_performance_train:', best_config_performance_train)
    print('')

    print('best_config_performance_test: ' + str(np.mean(best_config_performance_test['score'], axis=0)))
    print('best_config_performance_train: ' + str(np.mean(best_config_performance_train['score'], axis=0)))

    print('best_config_performance_test MAXIMUM: ' + str(np.max(np.mean(best_config_performance_test['score'], axis=0))))

    print('\nDone')
    # test = pandas.read_pickle(path=pre + 'Results//metrics_summary_test_oneHot')
    print('\nDone')
