import uuid
import numpy as np
from sklearn.datasets import load_breast_cancer

from photonai.processing.permutation_test import PermutationTest


def create_hyperpipe():
    # this is needed here for the parallelisation
    from photonai.base import Hyperpipe, PipelineElement, OutputSettings
    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import KFold

    settings = OutputSettings(mongodb_connect_url='mongodb://localhost:27017/photon_results')
    my_pipe = Hyperpipe('permutation_test_1',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=GroupKFold(n_splits=2),
                        inner_cv=KFold(n_splits=2),
                        calculate_metrics_across_folds=True,
                        use_test_set=True,
                        verbosity=1,
                        project_folder='./tmp/',
                        output_settings=settings)

    # Add transformer elements
    my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                               test_disabled=True, with_mean=True, with_std=True)

    my_pipe += PipelineElement("PCA", test_disabled=False)

    # Add estimator
    my_pipe += PipelineElement("SVC", hyperparameters={'kernel': ['linear', 'rbf']},
                               gamma='scale', max_iter=1000000)

    return my_pipe


X, y = load_breast_cancer(return_X_y=True)
my_perm_id = str(uuid.uuid4())
groups = np.random.random_integers(0, 3, (len(y), ))

# in case the permutation test for this specific hyperpipe has already been calculated, PHOTON will skip the permutation
# runs and load existing results
perm_tester = PermutationTest(create_hyperpipe, n_perms=2, n_processes=1, random_state=11,
                              permutation_id=my_perm_id)
perm_tester.fit(X, y, groups=groups)

results = PermutationTest._calculate_results(my_perm_id, mongodb_path='mongodb://localhost:27017/photon_results')
print(results.p_values)
