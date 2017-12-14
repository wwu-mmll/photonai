from DemoFiles.MultiTask.helpers import get_data
from DemoFiles.MultiTask.setup_model_MTL import setup_model_MTL
import numpy as np
import pandas

# ---------------------------
# run the analysis
if __name__ == '__main__':

    ###############################################################################################################
    #pre = 'C:/Users/hahnt/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/data_now_on_Titania/'
    pre = '/home/nils/data/GeneticBrainAtlas/'
    # pre = 'D:/myGoogleDrive/work/Papers/_underConstruction/BrainAtlasOfGeneticDepressionRisk/'

    group_id = 2    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    #target_modalities = ['custom_str', 'hip']
    #target_modalities = ['all']
    target_modalities = ['custom_id', 'Lhippo', 'Rhippo', 'lHip', 'rHip']

    one_hot_it = True

    getImportanceScores = False
    perm_test = False

    remove_covs = False
    covs = ['Alter', 'Geschlecht', 'Site', 'ICV']
    ###############################################################################################################

    # get data
    df, ROI_names, snp_names = get_data(pre=pre, one_hot_it=one_hot_it, what=target_modalities,
                                        impute_targets='mean')  # technically, we should always use drop here

    # Filter by diagnosis
    # 1=HC, 2=MDD, 3=BD, 4=Schizoaffective, 5=Schizophrenia, 6=other
    df = df.loc[df['Group'] == group_id]
    print('\n' + str(df.shape[0]) + ' samples remaining for Group ' + str(group_id) + '.')

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
        print('\nPERMUTATION TEST: SHUFFLING TARGETS NOW!')
        np.random.shuffle(targets)

    # remove confounders from target data (e.g. age, gender, site...)
    if remove_covs:
        print('\nRemoving covariates from targets.')
        import statsmodels.api as sm
        ols_X = df[covs]
        ols_X = sm.add_constant(ols_X)
        ols_model = sm.OLS(targets, ols_X)
        ols_results = ols_model.fit()
        targets = np.asarray(ols_results.resid)

    # scale targets
    print('\nScaling targets.\n')
    from sklearn.preprocessing import StandardScaler
    targets = StandardScaler().fit_transform(targets)

    # create PHOTON hyperpipe
    my_pipe, metrics = setup_model_MTL(target_info=target_info)
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

    print('\nDone')
    # test = pandas.read_pickle(path=pre + 'Results//metrics_summary_test_oneHot')
    print('\nDone')
