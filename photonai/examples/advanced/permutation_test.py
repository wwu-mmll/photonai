import uuid
from sklearn.datasets import load_breast_cancer

from photonai.processing import ResultsHandler
from photonai.processing.permutation_test import PermutationTest

def create_hyperpipe():
    # this is needed here for the unit tests
    from photonai.base import Hyperpipe, PipelineElement, OutputSettings
    from photonai.optimization import FloatRange, Categorical
    from sklearn.model_selection import KFold

    settings = OutputSettings(mongodb_connect_url='mongodb://trap-umbriel:27017/photon_results',
                              project_folder='./tmp/')
    my_pipe = Hyperpipe('basic_svm_pipe_permutation_test',
                        optimizer='sk_opt',
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
perm_tester = PermutationTest(create_hyperpipe, n_perms=20, n_processes=3, random_state=11,
                              permutation_id=str(uuid.uuid4()))
perm_tester.fit(X, y)

# Load results
handler = ResultsHandler()
handler.load_from_mongodb(mongodb_connect_url='mongodb://trap-umbriel:27017/photon_results',
                          pipe_name='basic_svm_pipe_permutation_test')

perm_results = handler.results.permutation_test
metric_dict = dict()
for metric in perm_results.metrics:
    metric_dict[metric.metric_name] = metric

p_acc = metric_dict['accuracy'].p_value
print('P value for metric accuracy: {}'.format(p_acc))


debug = True
