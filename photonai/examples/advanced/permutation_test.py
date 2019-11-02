import uuid
from sklearn.datasets import load_breast_cancer

from photonai.processing.permutation_test import PermutationTest


def create_hyperpipe():
    # this is needed here for the parallelisation
    from photonai.base import Hyperpipe, PipelineElement, OutputSettings
    from photonai.optimization import FloatRange, Categorical, IntegerRange
    from sklearn.model_selection import KFold

    settings = OutputSettings(mongodb_connect_url='mongodb://trap-umbriel:27017/photon_results',
                              project_folder='./tmp/')
    my_pipe = Hyperpipe('permutation_test_1',
                        optimizer='sk_opt',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=KFold(n_splits=2),
                        inner_cv=KFold(n_splits=2),
                        calculate_metrics_across_folds=True,
                        eval_final_performance=True,
                        verbosity=1,
                        output_settings=settings)

    # Add transformer elements
    my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                               test_disabled=True, with_mean=True, with_std=True)
    my_pipe += PipelineElement("PCA", hyperparameters={'n_components': IntegerRange(5, 15)},
                               test_disabled=False)

    # Add estimator
    my_pipe += PipelineElement("SVC", hyperparameters={'C': FloatRange(0.1, 5), 'kernel': ['linear', 'rbf']},
                               gamma='scale', max_iter=1000000)

    return my_pipe


X, y = load_breast_cancer(True)
my_perm_id = str(uuid.uuid4())

# in case the permutation test for this specific hyperpipe has already been calculated, PHOTON will skip the permutation
# runs and load existing results
perm_tester = PermutationTest(create_hyperpipe, n_perms=20, n_processes=1, random_state=11,
                              permutation_id=my_perm_id)
perm_tester.fit(X, y)

print(PermutationTest.get_permutation_status(my_perm_id, "mongodb://trap-umbriel:27017/photon_results"))
