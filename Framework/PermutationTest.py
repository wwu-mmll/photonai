import numpy as np
from multiprocessing import Pool

from .PhotonBase import Hyperpipe


class PermutationTest:

    def __init__(self, hyperpipe: Hyperpipe, metric: str, greater_is_better, n_perms=1000, n_cores=1,
                 results_to_mongodb=False, random_state=15):
        self.hyperpipe = hyperpipe
        self.X = hyperpipe.X
        self.y_true = hyperpipe.y
        self.n_perms = n_perms
        self.n_cores = n_cores
        self.random_state = random_state
        self.results_to_mongodb = results_to_mongodb
        self.metric = metric
        self.greater_is_better = greater_is_better

    def fit(self):
        np.random.seed(self.random_state)

        # Run with true labels
        true_pipe = self.hyperpipe
        true_pipe.fit(self.X, self.y_true)

        # collect test set performances and calculate mean
        n_outer_folds = len(true_pipe.result_tree.outer_folds)
        performance = list()
        for fold in range(n_outer_folds):
            performance.append(true_pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[self.metric])
        true_performance = np.mean(performance)

        # Compute permutations
        self.y_perm = list()
        for perm in range(self.n_perms):
            self.y_perm.append(np.random.permutation(self.y_true))

        # Run parallel pool
        pool = Pool(self.n_cores)
        perm_performances = pool.map(self.run_parallized_permutation, np.arange(1, self.n_perms))

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

    def run_parallized_permutation(self, perm):
        perm_pipe = self.hyperpipe
        perm_pipe.verbose = -1
        perm_pipe.name = self.hyperpipe.name + '_perm_' + str(perm)
        perm_pipe.fit(self.X, self.y_perm[perm])

        # collect test set predictions
        n_outer_folds = len(perm_pipe.result_tree.outer_folds)
        performance = list()
        for fold in range(n_outer_folds):
            performance.append(perm_pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[self.metric])
        mean_performance = np.mean(performance)
        return mean_performance

    def calculate_p(self, true_performance, perm_performances):
        if self.greater_is_better:
            return np.sum(true_performance > np.asarray(perm_performances))/self.n_perms
        else:
            return np.sum(true_performance < np.asarray(perm_performances))/self.n_perms