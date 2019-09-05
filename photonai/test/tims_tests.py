from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, Stack, Switch, Branch, DataFilter
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.optimization.SpeedHacks import DummyPerformance
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold, LeaveOneOut, LeaveOneGroupOut
from photonai.investigator.Investigator import Investigator
from sklearn.datasets import load_boston, load_breast_cancer, make_classification, make_regression
from itertools import product
import time, pprint
import numpy as np
import warnings
warnings.filterwarnings("ignore")   # mute warnings


def display_time(seconds, granularity=2):
    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    result = []
    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


def get_HP(pipe_name, project_folder, optimizer, optimizer_params, metrics, eval_final_performance, inner_cv, outer_cv,
           performance_constraints, plots_bool, cache_folder):
    return Hyperpipe(name=pipe_name,
                     output_settings=OutputSettings(project_folder=project_folder, plots=plots_bool),
                     optimizer=optimizer, optimizer_params=optimizer_params,
                     best_config_metric=metrics[0],
                     metrics=metrics,
                     inner_cv=inner_cv,
                     outer_cv=outer_cv,
                     eval_final_performance=eval_final_performance,
                     performance_constraints=performance_constraints,
                     cache_folder=cache_folder,
                     verbosity=-1)


def add_architecture(pipe: Hyperpipe, kind: str, case=1, show_investigator=False):
    # create random covariates for testing
    n_samples = 40
    groups = np.random.randint(low=1, high=3+1, size=n_samples)
    cov1 = np.random.rand(n_samples)
    cov2 = np.random.rand(n_samples)
    if kind is 'reg':
        # X, y = load_boston(True)    # GET DATA: Boston Housing REGRESSION example
        X, y = make_regression(n_samples=n_samples, n_features=20)  # generate data
        if case == 1:
            # just an SVM
            pipe += PipelineElement(name='SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf'])})
        elif case == 2:
            # Simple estimator Switch
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch
        elif case == 3:
            # estimator Switch without hyperparameters
            my_switch = Switch('estimator_switch')
            my_switch += PipelineElement('SVR')
            my_switch += PipelineElement('RandomForestRegressor')
            pipe += my_switch
        elif case == 4:
            # Transformer Switch
            my_switch = Switch('trans_switch')
            my_switch += PipelineElement('PCA')
            my_switch += PipelineElement('FRegressionSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += my_switch
            pipe += PipelineElement('RandomForestRegressor')
        elif case == 5:
            # multi-switch
            # setup switch to choose between PCA or simple feature selection and add it to the pipe
            pre_switch = Switch('preproc_switch')
            pre_switch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
            pre_switch += PipelineElement('FRegressionSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += pre_switch
            # setup estimator switch and add it to the pipe
            estimator_switch = Switch('estimator_switch')
            estimator_switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
            estimator_switch += PipelineElement('RandomForestRegressor', hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += estimator_switch
        elif case == 6:
            # Simple estimator Stack (use mean in the end)
            SVR = PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestRegressor', hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += Stack('estimator_stack', elements=[SVR, RF])
            pipe += PipelineElement('PhotonVotingRegressor')
        elif case == 7:
            # Simple estimator Stack, but use same machine twice
            SVR1 = PipelineElement('SVR',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVR2 = PipelineElement('SVR',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            pipe += Stack('estimator_stack', elements=[SVR1, SVR2])
            pipe += PipelineElement('PhotonVotingRegressor')
        elif case == 8:
            pipe += PipelineElement('StandardScaler')
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch
        elif case == 9:
            # sample pairing with confounder removal
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
            pipe += PipelineElement('SamplePairingRegression',
                                    {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                    distance_metric='euclidean', test_disabled=False)
            pipe += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                            'C': Categorical([.01, 1, 5])})
        elif case == 10:
            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingRegression', {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                               distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)
            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])
            # final estimator with stack output as features
            pipe += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})

        elif case == 11:
            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingRegression', {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                              distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVR', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestRegressor', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

    elif kind is 'clf':
        #X, y = load_breast_cancer(True)        # GET DATA: Breast Cancer CLASSIFICATION example
        X, y = make_classification(n_samples=n_samples, n_features=20)     # generate data
        if case == 1:
            # just an SVM
            pipe += PipelineElement(name='SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf'])})
        elif case == 2:
            # Simple estimator Switch
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch
        elif case == 3:
            # estimator Switch without hyperparameters
            my_switch = Switch('estimator_switch')
            my_switch += PipelineElement('SVC')
            my_switch += PipelineElement('RandomForestClassifier')
            pipe += my_switch
        elif case == 4:
            # Transformer Switch
            my_switch = Switch('trans_switch')
            my_switch += PipelineElement('PCA')
            my_switch += PipelineElement('FClassifSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += my_switch
            pipe += PipelineElement('RandomForestClassifier')
        elif case == 5:
            # multi-switch
            # setup switch to choose between PCA or simple feature selection and add it to the pipe
            pre_switch = Switch('preproc_switch')
            pre_switch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])}, test_disabled=True)
            pre_switch += PipelineElement('FClassifSelectPercentile', hyperparameters={'percentile': IntegerRange(start=5, step=20, stop=66, range_type='range')}, test_disabled=True)
            pipe += pre_switch
            # setup estimator switch and add it to the pipe
            estimator_switch = Switch('estimator_switch')
            estimator_switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
            estimator_switch += PipelineElement('RandomForestClassifier', hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += estimator_switch
        elif case == 6:
            # Simple estimator Stack (use mean in the end)
            SVR = PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestClassifier', hyperparameters={'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += Stack('estimator_stack', elements=[SVR, RF])
            pipe += PipelineElement('PhotonVotingClassifier')
        elif case == 7:
            # Simple estimator Stack, but use same machine twice
            SVC1 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVC2 = PipelineElement('SVC',
                                   hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            pipe += Stack('estimator_stack', elements=[SVC1, SVC2])
            pipe += PipelineElement('PhotonVotingClassifier')
        elif case == 8:
            pipe += PipelineElement('StandardScaler')
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch
        elif case == 9:
            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingClassification', {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                               distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)
            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])
            # final estimator with stack output as features
            pipe += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})

        elif case == 10:
            # crazy everything
            pipe += PipelineElement('StandardScaler')
            pipe += PipelineElement('SamplePairingClassification', {'draw_limit': [100], 'generator': Categorical(['nearest_pair', 'random_pair'])},
                                              distance_metric='euclidean', test_disabled=True)
            # setup pipeline branches with half of the features each
            # if both PCAs are disabled, features are simply concatenated and passed to the final estimator
            source1_branch = Branch('source1_features')
            # first half of features (for Boston Housing, same as indices=[0, 1, 2, 3, 4, 5]
            source1_branch += DataFilter(indices=np.arange(start=0, stop=int(np.floor(X.shape[1] / 2))))
            source1_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source1_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            source2_branch = Branch('source2_features')
            # second half of features (for Boston Housing, same is indices=[6, 7, 8, 9, 10, 11, 12]
            source2_branch += DataFilter(indices=np.arange(start=int(np.floor(X.shape[1] / 2)), stop=X.shape[1]))
            source2_branch += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, test_disabled=True,
                                              confounder_names=['cov1', 'cov2'])
            source2_branch += PipelineElement('PCA', hyperparameters={'n_components': Categorical([None, 5])},
                                              test_disabled=True)

            # setup source branches and stack their output (i.e. horizontal concatenation)
            pipe += Stack('source_stack', elements=[source1_branch, source2_branch])

            # final estimator with stack output as features
            # setup estimator switch and add it to the pipe
            switch = Switch('estimator_switch')
            switch += PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear', 'rbf']),
                                                              'C': Categorical([.01, 1, 5])})
            switch += PipelineElement('RandomForestClassifier', hyperparameters={
                'min_samples_split': FloatRange(start=.05, step=.1, stop=.26, range_type='range')})
            pipe += switch

        elif case == 11:
            # Simple estimator Stack (train Random Forest on estimator stack proba outputs)
            # ToDo: USE PROBA OUTPUTS!
            # create estimator stack
            SVC1 = PipelineElement('SVC', hyperparameters={'kernel': Categorical(['linear']), 'C': Categorical([.01, 1, 5])})
            SVC2 = PipelineElement('SVC', hyperparameters={'kernel': Categorical(['rbf']), 'C': Categorical([.01, 1, 5])})
            RF = PipelineElement('RandomForestClassifier')
            # add to pipe
            pipe += Stack('estimator_stack', elements=[SVC1, SVC2, RF])
            pipe += PipelineElement('RandomForestClassifier')

    results = pipe.fit(X, y, **{'groups': groups, 'cov1': cov1, 'cov2': cov2})
    if show_investigator:
        Investigator.show(pipe)
        debug = True
    return results


# DESIGN YOUR PIPELINE
hp_tests = 'simple'
#architecture_tests = list(np.arange(start=1, stop=11+1))
architecture_tests = [3]
show_investigator = True

pipe_name = 'test_pipe' + '_' + hp_tests
if hp_tests is 'all':
    project_folder_list = ['./test_output/tmp_rel/']    #, 'C:/Users/Tim/Google Drive/work/all_skripts/py_code/test/examples/test_output/tmp/' ]    # don't set path, relative path, absolute path
    optimizer_list = ['random_grid_search', 'sk_opt']   # ['grid_search', 'random_grid_search', 'sk_opt']  ToDo: optimizer_params = {}  integrate in optimizer list
    eval_final_performance_list = [True, False]
    inner_cv_list = [KFold(n_splits=3, shuffle=True), ShuffleSplit(n_splits=1, test_size=.2), LeaveOneOut()]
    outer_cv_list = [None, KFold(n_splits=3, shuffle=True), ShuffleSplit(n_splits=1, test_size=.25), LeaveOneOut()]
    performance_constraints_list = [None]   #, [DummyPerformance(metric='pearson_correlation', margin=.15)]]     # ToDo: add score/error tests
    plots_bool_list = [True]    # , False]
    cache_folder_list = [None, './test_output/cache/']
    kind_list = ['reg', 'clf']   # ['reg', 'clf', 'multi_clf']
elif hp_tests is 'simple':
    project_folder_list = ['./test_output/tmp_rel/']    #, 'C:/Users/Tim/Google Drive/work/all_skripts/py_code/test/examples/test_output/tmp/' ]    # don't set path, relative path, absolute path
    optimizer_list = ['random_grid_search']   # ['grid_search', 'random_grid_search', 'sk_opt']  ToDo: optimizer_params = {}  integrate in optimizer list
    eval_final_performance_list = [True]
    inner_cv_list = [KFold(n_splits=3, shuffle=True)]
    outer_cv_list = [ShuffleSplit(n_splits=1, test_size=.2)]    #[LeaveOneGroupOut()]
    performance_constraints_list = [None]   #, [DummyPerformance(metric='pearson_correlation', margin=.15)]]     # ToDo: add score/error tests
    plots_bool_list = [True]    # , False]
    cache_folder_list = [None]
    kind_list = ['reg']   # ['reg', 'clf', 'multi_clf']


# iterate over all HP setup combinations
tic = time.time()
i = 1
j = 1
f = 0
d = 0
all_combs = list(product(project_folder_list, optimizer_list, eval_final_performance_list, inner_cv_list, outer_cv_list,
                    performance_constraints_list, plots_bool_list, cache_folder_list, kind_list))


for project_folder, optimizer, eval_final_performance, inner_cv, outer_cv, performance_constraints, plots_bool, cache_folder, kind in all_combs:
    if kind == 'clf':
        metrics = ['balanced_accuracy', 'accuracy', 'f1_score']  # classification metrics
    elif kind == 'multi_clf':
        metrics = []  # multiclass classification metrics
    elif kind == 'reg':
        metrics = ['mean_absolute_error', 'mean_squared_error', 'pearson_correlation']  # regression metrics

    # add elements to Hyperpipe and fit
    for case_id in architecture_tests:
        print('Running Hyperpipe Config Test ' + str(i) + '/' + str(len(all_combs) * len(architecture_tests)) + '...',
              end=' ')

        # create Hyperpipe
        pipe = get_HP(pipe_name=pipe_name, project_folder=project_folder, optimizer=optimizer, optimizer_params={},
                      metrics=metrics, eval_final_performance=eval_final_performance, inner_cv=inner_cv,
                      outer_cv=outer_cv,
                      performance_constraints=performance_constraints, plots_bool=plots_bool, cache_folder=cache_folder)

        try:
            results = add_architecture(pipe=pipe, kind=kind, case=case_id, show_investigator=show_investigator)
            d += 1
            print('Done!')
        except Exception as e:
            f += 1
            print('Failed!')
            print(e)
            # print current config
            pprint.pprint(kind)
            pprint.pprint(case_id)
            pprint.pprint(all_combs[j-1])
            pprint.pprint(metrics)
            print('\n\n')
        i += 1
    j += 1

elapsed = time.time() - tic
print(str(d) + '/' + str(i-1) + ' tests done.')
print(str(f) + '/' + str(i-1) + ' tests failed.')
print('Test took ' + str(display_time(seconds=elapsed, granularity=4)))

# ToDo: Investigator Tests
#Investigator.show(pipe)
# Investigator.load_from_file('my_regression_example_pipe_1_basic_hyperpipe',
#                             r'C:\Users\Tim\Google Drive\work\all_skripts\py_code\test\examples\Regression\my_output_folder\my_regression_example_pipe_1_basic_hyperpipe_results_2019-09-03_11-23-34\photon_result_file.p')

# ToDo: Predict from saved Hyperpipe .photon Tests

# ToDo: Test permutationTest

