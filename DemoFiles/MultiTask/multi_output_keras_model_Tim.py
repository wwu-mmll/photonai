from DemoFiles.MultiTask.helpers import get_data, setup_model_MTL
import numpy as np
import pandas

# ---------------------------
# run the analysis
if __name__ == '__main__':
    pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    #pre = '/home/nils/data/GeneticBrainAtlas'
    # pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/'

    group_id = 2    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    one_hot_it = True
    target_modalities = ['vol']
    getImportanceScores = False
    perm_test = False

    remove_covs = True
    covs = ['Alter', 'Geschlecht', 'Site', 'ICV']
    ###############################################################################################################
    # get data
    df, ROI_names, snp_names = get_data(pre=pre, one_hot_it=one_hot_it, what=target_modalities,
                                        impute_targets='mean')  # technically, we should always use drop here

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    # group_id = 2
    df = df.loc[df['Group'] == group_id]
    print(str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

    # Prepare multi task targets
    targets = np.asarray(df[ROI_names])

    # get data (numeric snps)
    data = np.asarray(df[snp_names])
    data = data.copy(order='C')  # fixes an error (don't know why this is necessary)

    # create list of dictionaries that define target_info
    target_info = []
    for roi_name in ROI_names:
        output_node_dict = {'name': roi_name, 'target_dimension': 1, 'activation': 'linear',
                            'loss': 'mse', 'loss_weight': 1}
        target_info.append(output_node_dict)

    # shuffle targets if running a permutation test
    if perm_test:
        print('PERMUTATION TEST: SHUFFLING TARGETS NOW!')
        np.random.shuffle(targets)

    # remove confounders from target data (age, gender, site, ICV)
    if remove_covs:
        print('Removing covariates from targets.')
        import statsmodels.api as sm
        ols_X = df[covs]
        ols_X = sm.add_constant(ols_X)
        ols_model = sm.OLS(targets, ols_X)
        ols_results = ols_model.fit()
        targets = np.asarray(ols_results.resid)

    # create PHOTON hyperpipe
    my_pipe, metrics = setup_model_MTL(target_info=target_info)
    # fit PHOTON model
    import time
    millis1 = int(round(time.time()))
    results = my_pipe.fit(data, targets)
    millis2 = int(round(time.time()))
    print('Time (minutes): ' + str((millis2 - millis1) / 60))

    # # get results
    # results_tree = results.result_tree
    # metrics_summary_train = pandas.DataFrame()
    # metrics_summary_test = pandas.DataFrame()
    # importance_scores_summary = pandas.DataFrame()
    # for roi in results:
    #     te, tr, imp = roi
    #     metrics_summary_test = metrics_summary_test.append(te)
    #     metrics_summary_train = metrics_summary_train.append(tr)
    #     if getImportanceScores:
    #         importance_scores_summary = importance_scores_summary.append(imp)
    #
    # metrics_summary_test = metrics_summary_test.sort_values(by='variance_explained', axis=0, ascending=False)
    # metrics_summary_train = metrics_summary_train.sort_values(by='variance_explained', axis=0, ascending=False)
    #
    # # save metrics summary
    # if perm_test:
    #     metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test_MTL_perm2')
    #     metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train_MTL_perm2')
    # else:
    #     metrics_summary_test.to_pickle(path=pre + 'Results/metrics_summary_test_MTL')
    #     metrics_summary_train.to_pickle(path=pre + 'Results/metrics_summary_train_MTL')
    #
    # if getImportanceScores and ~perm_test:
    #     importance_scores_summary.to_pickle(path=pre + 'Results/importance_scores_summary_MTL')

    print('Done')
    # test = pandas.read_pickle(path=pre + 'Results//metrics_summary_test_oneHot')
    print('Done')


print('')