from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings
from photonai.optimization.Hyperparameters import FloatRange, Categorical
from photonai.validation.PermutationTest import PermutationTest
from photonai.validation.ResultsTreeHandler import ResultsHandler

from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer


def create_hyperpipe():
    settings = OutputSettings(mongodb_connect_url='mongodb://trap-umbriel:27017/photon_results')
    my_pipe = Hyperpipe('basic_svm_pipe_permutation_test_3',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=KFold(n_splits=2),
                        inner_cv=KFold(n_splits=2),
                        calculate_metrics_across_folds=True,
                        eval_final_performance=True,
                        verbosity=1,
                        output_settings=settings)

    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('SVC', {'kernel': Categorical(['linear']),
                                       'C': FloatRange(1, 2, "linspace", num=2)})
    return my_pipe

X, y = load_breast_cancer(True)

# in case the permutation test for this specific hyperpipe has already been calculated, PHOTON will skip the permutation
# runs and load existing results
perm_tester = PermutationTest(create_hyperpipe, n_perms=20, n_processes=3, random_state=11, permutation_id='basic_svm_permutation_example_3')
perm_tester.fit(X, y)

# Load results
handler = ResultsHandler()
handler.load_from_mongodb(mongodb_connect_url='mongodb://trap-umbriel:27017/photon_results', pipe_name='basic_svm_pipe_permutation_test_3')

perm_results = handler.results.permutation_test
metric_dict = dict()
for metric in perm_results.metrics:
    metric_dict[metric.metric_name] = metric

p_acc = metric_dict['accuracy'].p_value
print('P value for metric accuracy: {}'.format(p_acc))


debug = True