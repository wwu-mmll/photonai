from multiprocessing import Pool, Process, Queue, current_process
import queue
import datetime
import numpy as np
import os
from ..photonlogger.Logger import Logger
from ..validation.Validate import Scorer, OptimizerMetric
from ..base.PhotonBase import OutputSettings
from ..validation.ResultsDatabase import MDBPermutationResults, MDBPermutationMetrics, MDBHyperpipe

from pymodm.errors import DoesNotExist


class PermutationTest:

    def __init__(self, hyperpipe_constructor, permutation_id:str, n_perms=1000, n_processes=1, random_state=15):

        self.hyperpipe_constructor = hyperpipe_constructor
        self.pipe = self.hyperpipe_constructor()

        # we need a mongodb to collect the results!
        if not self.pipe.output_settings.mongodb_connect_url:
            raise ValueError("MongoDB Connection String must be given for permutation tests")

        self.n_perms = n_perms
        self.permutation_id = permutation_id
        self.mother_permutation_id = permutation_id + "_reference"
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

        # at first we do a reference optimization
        y_true = y

        # Run with true labels

        # Check if it already exists in DB
        try:
            existing_reference = MDBHyperpipe.objects.raw({'permutation_id': self.mother_permutation_id,
                                                           'computation_completed': True}).first()
            # check if all outer folds exist
            Logger().info("Found hyperpipe computation with true targets, skipping the optimization process with true targets")
        except DoesNotExist:
            # if we havent computed the reference value do it:
            Logger().info("Calculating Reference Values with true targets.")
            try:
                self.pipe.permutation_id = self.mother_permutation_id
                self.pipe.fit(X, y_true)
                self.pipe.result_tree.computation_completed = True
                self.pipe.result_tree.save()
            except Exception as e:
                if self.pipe.result_tree is not None:
                    self.pipe.result_tree.permutation_failed = str(e)
                    self.pipe.result_tree.save()

        # find how many permutations have been computed already:
        existing_permutations = MDBHyperpipe.objects.raw({'permutation_id': self.permutation_id,
                                                          'computation_completed': True}).count()

        # we do one more permutation than is left in case the last permutation runs broke, one for each parallel
        if existing_permutations > 0 and (self.n_perms - existing_permutations) > 0:
            n_perms_todo = self.n_perms - existing_permutations # + self.n_processes
        else:
            n_perms_todo = self.n_perms

        if existing_permutations < self.n_perms:

            # Compute permutations
            np.random.seed(self.random_state)
            y_perms = list()
            for perm in range(n_perms_todo):
                y_perms.append(np.random.permutation(y_true))

            Logger().info(str(n_perms_todo) + " permutation runs todo")
            # Run parallel pool
            self.run_parallelized_hyperpipes(y_perms, self.hyperpipe_constructor, X, self.permutation_id,
                                             skip=existing_permutations)

        self.calculate_results()

        return self

    def run_parallelized_hyperpipes(self, y_perms, hyperpipe_constructor, X, permutation_id, skip=0):

        job_list = Queue()

        for perm_run, y_perm in enumerate(y_perms):
            perm_run_skip = perm_run + skip
            job_list.put([hyperpipe_constructor, X, perm_run_skip, y_perm, permutation_id])

        processes = []

        for w in range(self.n_processes):
            p = Process(target=do_jobs, args=(job_list,))
            processes.append(p)
            p.start()

            # completing process
        for p in processes:
            p.join()

        # pool = Pool(processes=self.n_processes, maxtasksperchild=1)
        # for perm_run, y_perm in enumerate(y_perms):
        #     perm_run = perm_run + skip
        #     pool.apply_async(run_parallelized_permutation, args=(hyperpipe_constructor, X, perm_run, y_perm, permutation_id),
        #                      callback=self.collect_results)
        # pool.close()
        # pool.join()


    def calculate_results(self):

        mother_permutation = MDBHyperpipe.objects.raw({'permutation_id': self.mother_permutation_id,
                                                       'computation_completed': True}).first()
        all_permutations = MDBHyperpipe.objects.raw({'permutation_id': self.permutation_id,
                                                     'computation_completed': True})
        number_of_permutations = all_permutations.count()

        # collect true performance
        # collect test set performances and calculate mean
        n_outer_folds = len(mother_permutation.outer_folds)
        true_performance = dict()
        for _, metric in self.metrics.items():
            performance = list()
            for fold in range(n_outer_folds):
                performance.append(mother_permutation.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[metric['name']])
            true_performance[metric['name']] = np.mean(performance)

        # collect perm performances

        perm_performances_global = list()
        for index, perm_pipe in enumerate(all_permutations):
            try:
                # collect test set predictions
                n_outer_folds = len(perm_pipe.outer_folds)
                perm_performances = dict()
                for _, metric in self.metrics.items():
                    performance = list()
                    for fold in range(n_outer_folds):
                        performance.append(
                            perm_pipe.outer_folds[fold].best_config.inner_folds[-1].validation.metrics[
                                metric['name']])
                    perm_performances[metric['name']] = np.mean(performance)
                perm_performances['ind_perm'] = index
                perm_performances_global.append(perm_performances)
            except Exception as e:
                # we suspect that the task was killed during computation of this permutation
                Logger().error("Dismissed one permutation from calculation:")
                Logger().error(e)

        # Reorder results
        perm_perf_metrics = dict()
        for _, metric in self.metrics.items():
            perms = list()
            for i in range(len(perm_performances_global)):
                perms.append(perm_performances_global[i][metric['name']])
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
        mother_permutation.permutation_test = perm_results
        mother_permutation.save()

        return true_performance, perm_perf_metrics

    def collect_results(self, result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        Logger().info("Finished Permutation Run" + str(result))

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


def do_jobs(tasks_to_accomplish):
    while True:
        try:
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            hyperpipe_construct = task[0]
            X = task[1]
            perm_run = task[2]
            y_perm = task[3]
            permutation_id = task[4]

            print(os.getpid())
            print("Starting permutation " + str(perm_run) + " from process number " + current_process().name)
            run_parallelized_permutation(hyperpipe_construct, X, perm_run, y_perm, permutation_id)
    return True


def run_parallelized_permutation(hyperpipe_constructor, X, perm_run, y_perm, permutation_id):
    # Create new instance of hyperpipe and set all parameters
    perm_pipe = hyperpipe_constructor()
    perm_pipe._set_verbosity(-1)
    perm_pipe.name = perm_pipe.name + '_perm_' + str(perm_run)
    perm_pipe.permutation_id = permutation_id

    # print(y_perm)
    po = OutputSettings(mongodb_connect_url=perm_pipe.output_settings.mongodb_connect_url,
                        save_predictions='None', save_feature_importances='None', save_output=False)
    perm_pipe._set_persist_options(po)
    perm_pipe.calculate_metrics_across_folds = False
    try:
        # Fit hyperpipe
        print('Fitting permutation ' + str(perm_run) + ' ...')
        perm_pipe.fit(X, y_perm)
        perm_pipe.result_tree.computation_completed = True
        perm_pipe.result_tree.save()
        print('Finished permutation ' + str(perm_run) + ' ...')
    except Exception as e:
        if perm_pipe.result_tree is not None:
            perm_pipe.result_tree.permutation_failed = str(e)
            perm_pipe.result_tree.save()
            print('Failed permutation ' + str(perm_run) + ' ...')
    return perm_run












