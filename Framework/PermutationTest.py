import numpy as np
from multiprocessing import Pool
from copy import deepcopy
from sklearn.model_selection import ShuffleSplit
from pymodm import connect
from .PhotonBase import Hyperpipe


class PermutationTest:

    def __init__(self, hyperpipe_constructor, metric: str, greater_is_better, n_perms=1000, n_processes=1,
                 results_to_mongodb=False, random_state=15):

        self.hyperpipe_constructor = hyperpipe_constructor
        self.pipe = self.hyperpipe_constructor()
        self.n_perms = n_perms
        self.n_processes = n_processes
        self.random_state = random_state
        self.results_to_mongodb = results_to_mongodb
        self.metric = metric
        self.greater_is_better = greater_is_better


    def fit(self, X, y):
        np.random.seed(self.random_state)
        y_true = y

        # Run with true labels
        self.pipe.fit(X, y_true)

        # collect test set performances and calculate mean
        n_outer_folds = len(self.pipe.result_tree.outer_folds)
        performance = list()
        for fold in range(n_outer_folds):
            performance.append(self.pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[self.metric])
        true_performance = np.mean(performance)

        # Compute permutations
        y_perms = list()
        for perm in range(self.n_perms):
            y_perms.append(np.random.permutation(y_true))

        # Run parallel pool
        pool = Pool(processes=self.n_processes)
        perm_performances = [pool.apply(run_parallized_permutation, args=(self.hyperpipe_constructor, X, perm_run, y_perm, self.metric)) for perm_run, y_perm in enumerate(y_perms)]
        pool.close()

        # Calculate p-value
        p = self.calculate_p(true_performance=true_performance, perm_performances=perm_performances)

        # Print results
        print(""" 
        Done with permutations...
        
        Results Permutation Test
        ===============================================
            True Performance:
                Metric {} = {} 
                Greater is Better = {}
                p Value = {}

        """.format(self.metric, true_performance, self.greater_is_better, p))
        return {'p': p, 'true_performance': true_performance, 'perm_performances': perm_performances}

    def calculate_p(self, true_performance, perm_performances):
        if self.greater_is_better:
            return np.sum(true_performance < np.asarray(perm_performances))/self.n_perms
        else:
            return np.sum(true_performance > np.asarray(perm_performances))/self.n_perms



def run_parallized_permutation(hyperpipe_constructor, X, perm_run, y_perm, metric):
    connect("mongodb://localhost:27017/permutation_test_test")
    # Create new instance of hyperpipe and set all parameters
    perm_pipe = hyperpipe_constructor()
    perm_pipe.verbose = -1
    perm_pipe.name = perm_pipe.name + '_perm_' + str(perm_run)

    # Fit hyperpipe
    perm_pipe.fit(X, y_perm)

    # collect test set predictions
    n_outer_folds = len(perm_pipe.result_tree.outer_folds)
    performance = list()
    for fold in range(n_outer_folds):
        performance.append(perm_pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[metric])
    mean_performance = np.mean(performance)
    return mean_performance







