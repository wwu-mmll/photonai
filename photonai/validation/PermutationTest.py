from multiprocessing import Pool

import numpy as np
from ..photonlogger.Logger import Logger
from ..validation.Validate import Scorer, OptimizerMetric
from ..base.PhotonBase import OutputSettings
from ..validation.ResultsDatabase import MDBPermutationResults, MDBPermutationMetrics


class PermutationTest:

    def __init__(self, hyperpipe_constructor, n_perms=1000, n_processes=1,
                 random_state=15):

        self.hyperpipe_constructor = hyperpipe_constructor
        self.pipe = self.hyperpipe_constructor()
        self.n_perms = n_perms
        self.n_processes = n_processes
        self.random_state = random_state

        # Get all specified metrics
        self.metrics = dict()
        for metric in self.pipe.metrics:
            self.metrics[metric] = {'name': metric, 'greater_is_better': self.set_greater_is_better(metric)}
        best_config_metric = self.pipe.best_config_metric
        if best_config_metric not in self.metrics.keys():
            self.metrics[best_config_metric] = {'name': best_config_metric,
                                                'greater_is_better': self.set_greater_is_better(best_config_metric)}

    def fit(self, X, y):
        y_true = y

        # Run with true labels
        self.pipe.fit(X, y_true)

        # collect test set performances and calculate mean
        n_outer_folds = len(self.pipe.result_tree.outer_folds)

        true_performance = dict()
        for _, metric in self.metrics.items():
            performance = list()
            for fold in range(n_outer_folds):
                performance.append(self.pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[metric['name']])
            true_performance[metric['name']] = np.mean(performance)

        # Compute permutations
        np.random.seed(self.random_state)
        y_perms = list()
        for perm in range(self.n_perms):
            y_perms.append(np.random.permutation(y_true))

        # Run parallel pool
        self.perm_performances = [None]*self.n_perms
        self.run_parallelized_hyperpipes(y_perms, self.hyperpipe_constructor, X, self.metrics)

        # Reorder results
        perm_perf_metrics = dict()
        for _, metric in self.metrics.items():
            perms = list()
            for i in range(self.n_perms):
                perms.append(self.perm_performances[i][metric['name']])
            perm_perf_metrics[metric['name']] = perms

        # Calculate p-value
        p = self.calculate_p(true_performance=true_performance, perm_performances=perm_perf_metrics)
        p_text = dict()
        for _, metric in self.metrics.items():
            if p[metric['name']] == 0:
                p_text[metric['name']] = "p < {}".format(str(1/self.n_perms))
            else:
                p_text[metric['name']] = "p = {}".format(p[metric['name']])

        # Print results
        Logger().info("""
        Done with permutations...

        Results Permutation test
        ===============================================
        """)
        for _, metric in self.metrics.items():
            Logger().info("""
                Metric: {}
                True Performance: {}
                p Value: {}

            """.format(metric['name'], true_performance[metric['name']], p_text[metric['name']]))

        # Write results to results tree
        perm_results = MDBPermutationResults(n_perms=self.n_perms, random_state=self.random_state)
        results_all_metrics = list()
        for _, metric in self.metrics.items():
            perm_metrics = MDBPermutationMetrics(metric_name=metric['name'], p_value=p[metric['name']], metric_value=true_performance[metric['name']])
            perm_metrics.values_permutations = perm_perf_metrics[metric['name']]
            results_all_metrics.append(perm_metrics)
        perm_results.metrics = results_all_metrics
        self.pipe.result_tree.permutation_test = perm_results
        self.pipe.mongodb_writer.save(self.pipe.result_tree)

        return {'pipe': self.pipe, 'p': p, 'true_performance': true_performance, 'perm_performances': perm_perf_metrics}

    def run_parallelized_hyperpipes(self, y_perms, hyperpipe_constructor, X, metrics):
        pool = Pool(processes=self.n_processes)
        for perm_run, y_perm in enumerate(y_perms):
            pool.apply_async(run_parallelized_permutation, args=(hyperpipe_constructor, X, perm_run, y_perm, metrics),
                             callback=self.collect_results)
        pool.close()
        pool.join()

    def collect_results(self, result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        self.perm_performances[result['ind_perm']] = result


    def calculate_p(self, true_performance, perm_performances):
        p = dict()
        for _, metric in self.metrics.items():
            if metric['greater_is_better']:
                p[metric['name']] = np.sum(true_performance[metric['name']] < np.asarray(perm_performances[metric['name']]))/(self.n_perms + 1)
            else:
                p[metric['name']] = np.sum(true_performance[metric['name']] > np.asarray(perm_performances[metric['name']]))/(self.n_perms + 1)
        return p

    def set_greater_is_better(self, metric):
        """
        Set greater_is_better for metric
        :param string specifying metric
        """
        if metric == 'score':
            # if no specific metric was chosen, use default scoring method

            last_element = self.pipe.pipeline_elements[-1]
            if hasattr(last_element.base_element, '_estimator_type'):
                greater_is_better = True
            else:
                # Todo: better error checking?
                Logger().error('NotImplementedError: ' +
                               'No metric was chosen and last pipeline element does not specify ' +
                               'whether it is a classifier, regressor, transformer or ' +
                               'clusterer.')
                raise NotImplementedError('No metric was chosen and last pipeline element does not specify '
                                          'whether it is a classifier, regressor, transformer or '
                                          'clusterer.')
        else:
            greater_is_better = OptimizerMetric.greater_is_better_distinction(metric)
        return greater_is_better


def run_parallelized_permutation(hyperpipe_constructor, X, perm_run, y_perm, metrics):
    # Create new instance of hyperpipe and set all parameters
    perm_pipe = hyperpipe_constructor()
    perm_pipe._set_verbosity(-1)
    perm_pipe.name = perm_pipe.name + '_perm_' + str(perm_run)

    po = OutputSettings(mongodb_connect_url='', local_file='', log_filename='',
                        save_predictions='None', save_feature_importances='None')
    perm_pipe._set_persist_options(po)
    perm_pipe.calculate_metrics_across_folds = False

    # Fit hyperpipe
    print('Fitting permutation...')
    perm_pipe.fit(X, y_perm)

    # collect test set predictions
    n_outer_folds = len(perm_pipe.result_tree.outer_folds)

    perm_performances = dict()
    for _, metric in metrics.items():
        performance = list()
        for fold in range(n_outer_folds):
            performance.append(
                perm_pipe.result_tree.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[metric['name']])
        perm_performances[metric['name']] = np.mean(performance)
    perm_performances['ind_perm'] = perm_run
    return perm_performances









