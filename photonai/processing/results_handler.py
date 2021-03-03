import csv
import os
import pickle
import pprint
import warnings
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from bson import json_util

from prettytable import PrettyTable
from pymodm import connect
from pymongo import DESCENDING
from pymongo.errors import DocumentTooLarge
from scipy.stats import sem

from photonai.photonlogger.logger import logger
from photonai.helper.helper import print_double_metrics, print_metrics, print_estimator_metrics, print_config_list_table
from photonai.processing.metrics import Scorer
from photonai.processing.results_structure import MDBHyperpipe
from photonai.__init__ import __version__


class ResultsHandler:
    """
    Provides all functions that operate on calculated results.
    As IO for the results object the ResultsHandler
    is able to handle results on its own.

    """
    def __init__(self, results_object: MDBHyperpipe = None, output_settings=None):
        """
        Initialize the object.

        Parameters:
            results_object:
                All results are stored here.
                An initial setting is not necessary,
                because a later loading via file or MongoDB is possible.

            output_settings (OutputSettings):
                Setting for creation and storage of the results_object.

        """
        self.results = results_object
        self.output_settings = output_settings

    def load_from_file(self, results_file: str):
        """
        Read results_file from json into MDBHyperpipe object self.results.

        Parameters:
            results_file:
                Full path to json file.

        """
        self.results = MDBHyperpipe.from_document(json.load(open(results_file, 'r')))

    def load_from_mongodb(self, mongodb_connect_url: str, pipe_name: str):
        """
        Read results_file from MongoDB into MDBHyperpipe object self.results.

        Parameters:
            mongodb_connect_url:
                MongoDB connection string.

            pipe_name:
                Name of the stored hyperpipe.

        """
        connect(mongodb_connect_url, alias="photon_core")
        results = list(MDBHyperpipe.objects.raw({'name': pipe_name}))
        if len(results) == 1:
            self.results = results[0]
        elif len(results) > 1:
            self.results = MDBHyperpipe.objects.order_by([("computation_start_time", DESCENDING)]).raw({'name': pipe_name}).first()
            warn_text = 'Found multiple hyperpipes with that name. Returning most recent one.'
            logger.warning(warn_text)
            warnings.warn(warn_text)
        else:
            raise FileNotFoundError('Could not load hyperpipe from MongoDB.')

    @staticmethod
    def get_methods() -> list:
        """
        This function returns a list of all methods available for ResultsHandler.

        Returns:
            List of all available methods.

        """
        methods_list = [s for s in dir(ResultsHandler) if '__' not in s]
        return methods_list

    def get_performance_table(self):
        """This function returns a summary table of the overall results.

        ToDo: add best_config information!
        """
        res_tab = pd.DataFrame()
        for i, folds in enumerate(self.results.outer_folds):
            # add best config infos
            res_tab.loc[i, 'best_config'] = str(folds.best_config.human_readable_config)

            # add fold index
            res_tab.loc[i, 'fold'] = folds.fold_nr

            # add sample size infos
            res_tab.loc[i, 'n_train'] = folds.best_config.best_config_score.number_samples_training
            res_tab.loc[i, 'n_validation'] = folds.best_config.best_config_score.number_samples_validation

            # add performance metrics
            d = folds.best_config.best_config_score.validation.metrics
            for key, value in d.items():
                res_tab.loc[i, key] = value

        # add row with overall info
        res_tab.loc[i + 1, 'n_validation'] = np.sum(res_tab['n_validation'])
        for key, value in d.items():
            m = res_tab.loc[:, key]
            res_tab.loc[i+1, key] = np.mean(m)
            res_tab.loc[i + 1, key + '_sem'] = sem(m)   # standard error of the mean
        res_tab.loc[i + 1, 'best_config'] = 'Overall'
        return res_tab

    def get_performance_outer_folds(self):
        performances = dict()
        for metric in self.results.outer_folds[0].best_config.best_config_score.validation.metrics.keys():
            performances[metric] = list()
        for i, fold in enumerate(self.results.outer_folds):
            for metric, value in fold.best_config.best_config_score.validation.metrics.items():
                performances[metric].append(value)
        return performances

    def get_config_evaluations(self) -> dict:
        """
        Return the test performance of every tested configuration in every outer fold.

        Returns:
            Test performance of every configuration.

        """
        config_performances = list()
        maximum_fold = None
        for outer_fold in self.results.outer_folds:
            if maximum_fold is None or len(outer_fold.tested_config_list) > maximum_fold:
                maximum_fold = len(outer_fold.tested_config_list)

        for outer_fold in self.results.outer_folds:
            performance = dict()
            for metric in self.results.hyperpipe_info.metrics:
                performance[metric] = list()

            for i in range(maximum_fold):
                # for config in outer_fold.tested_config_list:
                for metric in self.results.hyperpipe_info.metrics:
                    if i >= len(outer_fold.tested_config_list):
                        performance[metric].append(np.nan)
                        continue
                    config = outer_fold.tested_config_list[i]
                    if config.config_failed:
                        performance[metric].append(np.nan)
                    else:
                        for item in config.metrics_test:
                            if (item.operation == 'mean') and (item.metric_name == metric):
                                performance[metric].append(item.value)
            config_performances.append(performance)

        config_performances_dict = dict()
        for metric in self.results.hyperpipe_info.metrics:
            config_performances_dict[metric] = list()
            for fold in config_performances:
                config_performances_dict[metric].append(fold[metric])

        return config_performances_dict

    def get_minimum_config_evaluations(self):
        config_evaluations = self.get_config_evaluations()
        minimum_config_evaluations = dict()

        for metric, evaluations in config_evaluations.items():
            minimum_config_evaluations[metric] = list()
            greater_is_better = Scorer.greater_is_better_distinction(metric)

            for fold in evaluations:
                fold_evaluations = list()

                if greater_is_better:
                    for i, config in enumerate(fold):
                        if i == 0:
                            last_config = config
                        else:
                            if config > last_config:
                                last_config = config
                        fold_evaluations.append(last_config)
                else:
                    last_config = np.inf
                    for i, config in enumerate(fold):
                        if i == 0:
                            last_config = config
                        else:
                            if config < last_config:
                                last_config = config
                        fold_evaluations.append(last_config)
                minimum_config_evaluations[metric].append(fold_evaluations)

        return minimum_config_evaluations

    def get_learning_curves(self, config_nr, outer_fold_nr, save):
        """This function gets the learning curves out of the result tree.
        It returns the learning curves as a pandas dataframe.
        If save = True it saves the learning curves as a csv file.

        """
        cuts = self.results.hyperpipe_info.learning_curves_cut.values[1:] + [1.]
        fold_num = len(self.results.outer_folds[0].tested_config_list[config_nr - 1].inner_folds)
        idx = pd.MultiIndex.from_product([cuts, [i + 1 for i in range(fold_num)]], names=['Cut', 'Inner Fold Nr.'])
        col = pd.MultiIndex.from_product([self.results.hyperpipe_info.metrics, ['test', 'train']])
        data = {}
        for metric in self.results.hyperpipe_info.metrics:
            config = self.results.outer_folds[outer_fold_nr - 1].tested_config_list[config_nr - 1]
            for t in [1, 2]:
                curves = []
                for cut_nr, cut in enumerate(cuts):
                    curves += [config.inner_folds[fold].learning_curves[cut_nr][t][metric] for fold in range(fold_num)]
                data.update({(metric, ['test', 'train'][t-1]): curves})
        curves = pd.DataFrame(data, index=idx, columns=col)
        if save:
            curves.to_csv(self._save_prep_learning_curves('lc_outer_fold_%d_config_%d.csv' % (outer_fold_nr, config_nr)))
        return curves

    def plot_curves(self, curves: pd.DataFrame, title: str = 'Learning Curves'):
        """This function plots the learning curves.

        Parameters:
            curves:
                Dataframe with multi-index: (run - fraction of data)
                columns: at least (metric, train/test) floats

            title:
                Subtitle of plot.

        """
        metrics = self.results.hyperpipe_info.metrics
        fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 4., 4.))
        if len(metrics) == 1:
            axes = [axes]
        cuts = curves.index.get_level_values(0)
        col_template = tuple(curves.columns[0])
        # iterate only over first 2 entries [(metric, train/test)], example 'mean' as third

        for metric, ax in zip(metrics, axes):
            for subset in ['test', 'train']:
                sns.lineplot(x=cuts, y=curves[(metric, subset)+col_template[2:]], label=metric + '_' + subset, ax=ax)
            ax.set(xlabel='Fraction of Train Data used', ylabel='Metric Value')
            ax.legend(fontsize='small')
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def plot_learning_curves_config(self, config_nr, outer_fold_nr, save, show=False):
        """This function gets the learning curves for a specific config nr. and outer fold nr. and plots them
        If config_nr = -1 it gets the best config of the outer fold
        If save = True the plot is saved
        If show = True the plot is shown
        """
        if config_nr == -1:
            config_nr = self.results.best_config.config_nr
        curves = self.get_learning_curves(config_nr, outer_fold_nr, save)
        curves.columns = curves.columns.to_flat_index()
        fig = self.plot_curves(curves, 'Learning Curves (Outer Fold Nr.%d Config Nr.%d)' % (outer_fold_nr, config_nr))
        if save:
            plt.savefig(self._save_prep_learning_curves('lc_outer_fold_%d_config_%d.png' % (outer_fold_nr, config_nr)))
        if show:
            plt.show()
        plt.close()

    def plot_learning_curves_outer_fold(self, outer_fold_nr, config_nr_list=None, save=True, show=False):
        """This function gets the learning curves for a list of configs in a specific outer fold and plots them
        For each config the mean of the learning curves of all inner folds is used
        If config_nr = -1 it gets the best config of the outer fold
        If save = True the plot is saved
        If show = True the plot is shown
        """
        if config_nr_list is None:
            config_nr_list = np.arange(1, len(self.results.outer_folds[outer_fold_nr - 1].tested_config_list) + 1)
        elif -1 in config_nr_list:
            config_nr_list = [nr for nr in config_nr_list if nr is not self.results.best_config.config_nr]
            config_nr_list[config_nr_list == -1] = self.results.best_config.config_nr
        curves_list = []
        for config_nr in config_nr_list:
            curves = self.get_learning_curves(config_nr, outer_fold_nr, save)
            curves_list.append(curves.groupby(level=0).agg(['mean']))
        curves_configs = pd.concat(curves_list, axis=0, names=["Config Nr."], keys=config_nr_list)
        curves_configs.columns = curves_configs.columns.to_flat_index()
        curves_configs = curves_configs.swaplevel()
        fig = self.plot_curves(curves_configs, 'Learning Curves (Outer Fold Nr.%d)' % outer_fold_nr)
        if save:
            curves_configs.to_csv(self._save_prep_learning_curves('lc_outer_fold_{}.csv'.format(outer_fold_nr)))
            plt.savefig(self._save_prep_learning_curves('lc_outer_fold_{}.png'.format(outer_fold_nr)))
        if show:
            plt.show()
        plt.close()

    def _save_prep_learning_curves(self, file_name):
        path = self.results.output_folder + '/learning_curves/'
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(path, file_name)

    def save_all_learning_curves(self):
        for outer_fold_nr in range(1, len(self.results.outer_folds) + 1):
            for config_nr in range(1, len(self.results.outer_folds[0].tested_config_list) + 1):
                self.plot_learning_curves_config(config_nr, outer_fold_nr, save=True)

    def plot_optimizer_history(self, metric,
                               title: str = 'Optimizer History',
                               type: str = 'plot',
                               reduce_scatter_by: Union[int, str] = 'auto',
                               file: str = None):
        """
        :param metric: specify metric that has been stored within the PHOTONAI results tree
        :param type: 'plot' or 'scatter'
        :param reduce_scatter_by: integer or string ('auto'), reduce the number of points plotted by scatter
        :param file: specify a filename if you want to save the plot
        :return:
        """

        if metric not in self.results.hyperpipe_info.metrics:
            raise ValueError('Metric "{}" not stored in results tree'.format(metric))

        config_evaluations = self.get_config_evaluations()
        minimum_config_evaluations = self.get_minimum_config_evaluations()

        # handle different lengths
        min_corresponding = len(min(config_evaluations[metric], key=len))
        config_evaluations_corres = [configs[:min_corresponding] for configs in config_evaluations[metric]]
        minimum_config_evaluations_corres = [configs[:min_corresponding]
                                             for configs in minimum_config_evaluations[metric]]

        mean = np.nanmean(np.asarray(config_evaluations_corres), axis=0)
        mean_min = np.nanmean(np.asarray(minimum_config_evaluations_corres), axis=0)

        greater_is_better = Scorer.greater_is_better_distinction(metric)
        if greater_is_better:
            caption = 'Maximum'
        else:
            caption = 'Minimum'

        plt.figure()
        if type == 'plot':
            plt.plot(np.arange(0, len(mean)), mean, '-', color='gray', label='Mean Performance')

        elif type == 'scatter':
            # now do smoothing
            if isinstance(reduce_scatter_by, str):
                if reduce_scatter_by != 'auto':
                    msg = '{} is not a valid smoothing_kernel specifier. ' \
                          'Falling back to "auto".'.format(reduce_scatter_by)
                    logger.warning(msg)
                    warnings.warn(msg)

                # if auto, then calculate size of reduce_scatter_by so that 75 points on x remain
                # smallest reduce_scatter_by should be 1
                reduce_scatter_by = max([np.floor(min_corresponding / 75).astype(int), 1])

            if reduce_scatter_by > 1:
                plt.plot([], [], ' ', label="scatter reduced by factor {}".format(reduce_scatter_by))

            for i, fold in enumerate(config_evaluations[metric]):
                # add a few None so that list can be divided by smoothing_kernel
                remaining = len(fold) % reduce_scatter_by
                if remaining:
                    fold.extend([np.nan] * (reduce_scatter_by - remaining))
                # calculate mean over every n named_steps so that plot is less cluttered
                reduced_fold = np.nanmean(np.asarray(fold).reshape(-1, reduce_scatter_by), axis=1)
                reduced_xfit = np.arange(reduce_scatter_by / 2, len(fold), step=reduce_scatter_by)
                if i == len(config_evaluations[metric])-1:
                    plt.scatter(reduced_xfit, np.asarray(reduced_fold), color='gray', alpha=0.5, label='Performance', marker='.')
                else:
                    plt.scatter(reduced_xfit, np.asarray(reduced_fold), color='gray', alpha=0.5, marker='.')
        else:
            raise ValueError('Please specify either "plot" or "scatter".')

        plt.plot(np.arange(0, len(mean_min)), mean_min, '-', color='black', label='Mean {} Performance'.format(caption))

        for i, fold in enumerate(minimum_config_evaluations[metric]):
            xfit = np.arange(0, len(fold))
            plt.plot(xfit, fold, '-', color='black', alpha=0.5)

        plt.ylabel(metric.replace('_', ' '))
        plt.xlabel('No of Evaluations')

        plt.legend()
        plt.title(title)
        if file:
            plt.savefig(file)
        else:
            file = os.path.join(self.results.output_folder, "optimizer_history.png")
            plt.savefig(file)
        plt.close()

    def get_importance_scores(self):
        """
        This function returns the importance scores for the best configuration of each outer fold.
        """
        imps = []
        for i, fold in enumerate(self.results.outer_folds):
            imps.append(fold.best_config.best_config_score.feature_importances)
        return imps

    @staticmethod
    def collect_fold_lists(score_info_list, fold_nr, predictions_filename=''):
        if len(score_info_list) > 0:
            fold_nr_array = []
            collectables = {'y_pred': [], 'y_true': [], 'indices': [], 'probabilities': []}

            for i, score_info in enumerate(score_info_list):
                for collectable_key, collectable_list in collectables.items():
                    if getattr(score_info, collectable_key) is not None and len(
                            getattr(score_info, collectable_key)) > 0:
                        collectables[collectable_key].extend(list(getattr(score_info, collectable_key)))
                    else:
                        collectables[collectable_key].extend(list(np.full((len(score_info.y_true)), np.nan)))
                fold_nr_array.extend(list(np.ones((len(score_info.y_true),)) * fold_nr[i]))

            # enable nd y_pred support
            if len(collectables["y_pred"]) > len(collectables["y_true"]):
                tmp_collectables_y_pred = collectables["y_pred"]
                headers = collectables["y_pred"][0]
                for i, header in enumerate(list(headers)):
                    collectables[header] = [x[i] for x in tmp_collectables_y_pred if x != tmp_collectables_y_pred[0]]

            collectables["fold"] = fold_nr_array
            # convert to pandas dataframe to use their sorting algorithm
            save_df = pd.DataFrame(collectables)
            sorted_df = save_df.sort_values(by='indices')

            if predictions_filename != '':
                sorted_df.to_csv(predictions_filename, index=None)

            return sorted_df.to_dict('list')

    def get_test_predictions(self, filename=''):
        """
        This function returns the predictions, true targets, and fold index
        for the best configuration of each outer fold.
        """
        if self.results is None:
            raise ValueError("Result tree information is needed but results attribute of object is None.")

        score_info_list = list()
        fold_nr_list = list()
        for outer_fold in self.results.outer_folds:
            score_info_list.append(outer_fold.best_config.best_config_score.validation)
            fold_nr_list.append(outer_fold.fold_nr)
        return self.collect_fold_lists(score_info_list, fold_nr_list, filename)

    def get_validation_predictions(self, outer_fold_nr=0, config_no=0, filename=''):
        """
        This function returns the predictions, probabilities, true targets, fold and index
        for the config_nr of the given outer_fold
        """
        score_info_list = list()
        fold_nr_list = list()

        if self.results is None:
            raise ValueError("Result tree information is needed but results attribute of object is None.")

        # Todo: find config by config_id
        for inner_fold in self.results.outer_folds[outer_fold_nr].tested_config_list[config_no].inner_folds:
            score_info_list.append(inner_fold.validation)
            fold_nr_list.append(inner_fold.fold_nr)

        return self.collect_fold_lists(score_info_list, fold_nr_list, filename)

    def eval_mean_time_components(self, write_results=True, plotly_return=False):
        """
            This function create charts and tables out of the time-monitoring.
        """
        result_dict = {}
        caching = False
        default_dict = {'total_seconds': 0,
                        'total_items_processed': 0,
                        'mean_seconds_per_config': 0,
                        'mean_seconds_per_item': 0}

        # sum up times per element, 1. per config, and 2. in total
        for outer_fold in self.results.outer_folds:
            for config_nr, config in enumerate(outer_fold.tested_config_list):
                tmp_config_dict = {}
                # resort time entries for each element so that is has the following structure
                # element_name -> fit/transform/predict -> (seconds, nr_items)
                for inner_fold in config.inner_folds:
                    for time_key, time_values in inner_fold.time_monitor.items():
                        for value_item in time_values:
                            name, time, nr_items = value_item[0], value_item[1], value_item[2]
                            if name not in tmp_config_dict:
                                tmp_config_dict[name] = {}
                            if time_key not in tmp_config_dict[name]:
                                tmp_config_dict[name][time_key] = []
                            tmp_config_dict[name][time_key].append((time, nr_items))

                # calculate mean time per config and absolute time
                for element_name, element_time_dict in tmp_config_dict.items():
                    for element_time_key, element_time_list in element_time_dict.items():
                        if element_time_key == "transform_cached":
                            caching = True
                        mean_time = np.mean([i[0] for i in element_time_list])
                        total_time = np.sum([i[0] for i in element_time_list])
                        total_items_processed = np.sum([i[1] for i in element_time_list])

                        if element_name not in result_dict:
                            result_dict[element_name] = {}
                        if element_time_key not in result_dict[element_name]:
                            result_dict[element_name][element_time_key] = dict(default_dict)

                        result_dict[element_name][element_time_key]['total_seconds'] += total_time
                        result_dict[element_name][element_time_key]['total_items_processed'] += total_items_processed
                        mean_time_per_config = result_dict[element_name][element_time_key]['mean_seconds_per_config']
                        tmp_total_mean = ((mean_time_per_config * config_nr) + mean_time) / (config_nr + 1)
                        result_dict[element_name][element_time_key]['mean_seconds_per_config'] = tmp_total_mean
                        tmp_mean_per_item = result_dict[element_name][element_time_key]['total_seconds'] / \
                                            result_dict[element_name][element_time_key]['total_items_processed']
                        result_dict[element_name][element_time_key]['mean_seconds_per_item'] = tmp_mean_per_item

        format_str = '{:06.6f}'
        if caching:
            # in case we used caching add transform_cached and transform_computed values to transform_total
            for name, sub_result_dict in result_dict.items():
                if "transform_cached" in sub_result_dict:
                    result_dict[name]["transform"] = dict(default_dict)
                    for value_dict in sub_result_dict.values():
                        for info in value_dict.keys():
                            result_dict[name]["transform"][info] = result_dict[name]["transform_cached"][info]
                            # in case everything's been in the cache we have no computation
                            if "transform_computed" in sub_result_dict:
                                result_dict[name]["transform"][info] += result_dict[name]["transform_computed"][info]
                    if "transform_computed" in sub_result_dict:
                        # calculate a ratio, if caching was helpful and how much of the time it saved
                        result_dict[name]["cache_ratio"] = result_dict[name]["transform_cached"]["total_seconds"] / \
                                                           result_dict[name]["transform_computed"]["total_seconds"]

            # in case of caching we have different plot plus a different csv file
            csv_keys = ["fit", "transform", "transform_computed", "transform_cached", "predict"]
            csv_titles = csv_keys
            plot_list = ["fit", "transform", "transform_cached"]
            method_list = ["fit", "transform_computed", "transform_cached", "predict"]
        else:
            csv_keys = ["fit", "transform_computed", "predict"]
            csv_titles = ["fit", "transform", "predict"]
            plot_list = ["fit", "transform_computed"]
            method_list = ["fit", "transform_computed", "predict"]

        # write csv file with time analysis
        if write_results:
            sub_keys = ["total_seconds", "mean_seconds_per_config", "mean_seconds_per_item"]
            csv_filename = os.path.join(self.results.output_folder, 'time_monitor.csv')
            with open(csv_filename, 'w') as csvfile:
                writer = csv.writer(csvfile)
                header1 = [""]
                for k_name in csv_titles:
                    header1.extend([k_name, "", ""])
                header2 = ["Element"] + (sub_keys * len(csv_titles))
                if caching:
                    header1.append("")
                    header2.append("cache_ratio")
                writer.writerow(header1)
                writer.writerow(header2)
                for item, item_dict in result_dict.items():
                    row = [item]
                    for time_key in csv_keys:
                        for sub_key in sub_keys:
                            if time_key in item_dict:
                                row.append(format_str.format(item_dict[time_key][sub_key]))
                            else:
                                row.append('')
                    if caching:
                        if "cache_ratio" in item_dict:
                            row.append(item_dict["cache_ratio"])
                    writer.writerow(row)

        # plot figure
        # TODO! Use PiePlotlyPlot class without cricle imports
        plotly_dict = {'layout': {'title': 'Time Monitor Pie Chart',
                                  'showlegend': True,
                                  'height': 600,
                                  'annotations': []},
                       'data': []
                       }

        def append_plotly(labels, values, name, colors, domain):
            """
            helper function (temporary -> to.do above)
            """
            plotly_dict["data"].append({'labels': labels,
                                        'values': values,
                                        'type': 'pie',
                                        'name': name,
                                        'marker': {'colors': colors},
                                        'domain': domain,
                                        'hoverinfo': 'label+percent',
                                        'textposition': 'inside'})
            plotly_dict['layout']['annotations'].append({
                "x": np.mean(domain["x"]),
                "y": (domain["y"][1]),
                "font": {
                    "size": 16
                },
                "text": name,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False
            })

        def eval_mean_time_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                if pct/total >= 1:
                    return str(round(pct, 1))+"%"
                else:
                    return None

            return my_autopct

        # Create nxm sub plots
        cpl = len(plot_list)
        gs = matplotlib.gridspec.GridSpec(int((cpl-1)/3)+2, min(cpl, 3))
        legend_theme = plt.get_cmap('Set3')
        legend_theme2 = plt.get_cmap('tab10')

        element_names = [name for name, element in result_dict.items()]

        fig = plt.figure(figsize=(10, 7), dpi=160)
        colors = [legend_theme(1. * i / len(element_names)) for i in range(len(element_names))]
        for i, k in enumerate(plot_list):
            ax = plt.subplot(gs[int(i/3), i % 3])
            ax.set_prop_cycle("color", colors)
            data = [element[k]["total_seconds"] if k in element else 0 for name, element in result_dict.items()]
            data_sum = sum(data)
            if data_sum == 0:
                data_sum = 1
            values = [val/data_sum for val in data]
            patches, _, _ = plt.pie(values,
                                    shadow=True,
                                    startangle=90,
                                    autopct=eval_mean_time_autopct(data),
                                    pctdistance=0.7)
            plt.axis('equal')
            plt.title(k)
            append_plotly(labels=[str(d) for d in element_names],
                          values=values,
                          name=k,
                          colors=colors,
                          domain={'x': [i/len(plot_list), (i+1)/len(plot_list)], 'y': [0.55, 1]})

        plt.legend(
            loc='upper left',
            labels=['%s' % l for l in element_names],
            prop={'size': 10},
            bbox_to_anchor=(0.0, 1),
            bbox_transform=fig.transFigure
        )

        # add another plot for the comparison of the fit/transform/predict methods
        ax2 = plt.subplot(gs[int(i/3)+1, :])
        colors = [legend_theme2(1. * i / len(data)) for i in range(len(method_list))]
        ax2.set_prop_cycle("color", colors)
        data = []
        for k in method_list:
            data.append(np.sum([element[k]["total_seconds"] for name, element in result_dict.items() if k in element]))
        patches_an, _, _ = plt.pie([val/sum(data) for val in data],
                                   shadow=True,
                                   startangle=90,
                                   pctdistance=0.7,
                                   autopct=eval_mean_time_autopct(data))

        append_plotly(labels=method_list, values=[val / sum(data) for val in data], name="methods",
                      colors=colors, domain={'x': [0, 1], 'y': [0, 0.45]})

        plt.axis('equal')
        plt.title("methods")
        plt.legend(
            loc='lower left',
            labels=['%s' % l for l in method_list],
            prop={'size': 10},
            bbox_transform=fig.transFigure
        )

        # for only one legend
        #fig.legend(patches+patches_an, element_names+method_list, prop={'size': 10}, loc='lower left')

        if write_results:
            plt.savefig(os.path.join(self.results.output_folder, 'time_monitor_pie.png'))
        plt.close()
        if plotly_return:
            str_fig = "var layout =" + str(plotly_dict["layout"]) + ";"
            str_fig += "var data = " + str(plotly_dict["data"]) + ";"
            str_fig += "Plotly.newPlot('" + "time_monitor_pie_id" + "',data, layout);"
            return str_fig.replace("False", "false").replace("True", "true")

    def save(self):

        if self.output_settings.mongodb_connect_url:
            connect(self.output_settings.mongodb_connect_url, alias='photon_core')
            logger.info('Write results to mongodb...')
            try:
                self.results.save()
            except DocumentTooLarge:
                logger.error('Could not save document into MongoDB: Document too large')
        if self.output_settings.save_output:
            logger.info("Writing results to project folder...")
            self.write_result_tree_to_file()

    def save_backmapping(self, filename: str, backmapping):
        try:
            if isinstance(backmapping, list):
                backmapping = np.asarray(backmapping)

            try:
                from nibabel.nifti1 import Nifti1Image
                if isinstance(backmapping, Nifti1Image):
                    backmapping.to_filename(os.path.join(self.results.output_folder, filename + '.nii.gz'))
            except ImportError:
                pass
            finally:
                if isinstance(backmapping, np.ndarray):
                    if backmapping.size > 1000:
                        np.savez(os.path.join(self.results.output_folder, filename + '.npz'), backmapping)
                    else:
                        np.savetxt(os.path.join(self.results.output_folder, filename + '.csv'), backmapping, delimiter=',')
                else:
                    with open(os.path.join(self.results.output_folder, filename + '.p'), 'wb') as f:
                        pickle.dump(backmapping, f)
        except Exception as e:
            logger.error("Could not save backmapped feature importances.")
            logger.error(e)

    def write_convenience_files(self):
        if self.output_settings.save_output:
            logger.info("Writing summary file, plots and prediction csv to result folder ...")
            self.write_summary()
            self.write_predictions_file()

    def convert_to_json_serializable(self, value):
        if isinstance(value, (np.int, np.int32, np.int64)):
            return int(value)
        if isinstance(value, (np.float, np.float32, np.float64)):
            if self.output_settings.reduce_space:
                return round(float(value), 3)
            return float(value)
        else:
            return json_util.default(value)

    def write_result_tree_to_file(self):
        try:
            local_file = os.path.join(self.results.output_folder, 'photon_result_file.json')
            result = self.round_floats(self.results.to_son().to_dict())

            with open(local_file, 'w') as outfile:
                json.dump(result, outfile, default=self.convert_to_json_serializable)
        except OSError as e:
            logger.error("Could not write results to local file")
            logger.error(str(e))

    @classmethod
    def round_floats(cls, d):
        # recursive method for rounding all floats in result.json
        result = {}
        if isinstance(d, dict):
            for key, value in d.items():
                value = cls.round_floats(value)
                result.update({key: value})
            return result
        elif isinstance(d, list):
            return [cls.round_floats(val) for val in d]
        elif isinstance(d, float):
            return round(d, 6)
        else:
            return d

    def get_best_config_inner_fold_predictions(self, filename=''):
        score_info_list = []
        fold_nr = []
        for inner_fold in self.results.best_config.inner_folds:
            score_info_list.append(inner_fold.validation)
            fold_nr.append(inner_fold.fold_nr)
        return self.collect_fold_lists(score_info_list, fold_nr, filename)

    def write_predictions_file(self):
          if self.output_settings.save_output:
            filename = os.path.join(self.output_settings.results_folder, 'best_config_predictions.csv')
            # usually we write the predictions for the outer fold
            if not self.output_settings.save_predictions_from_best_config_inner_folds:
                return self.get_test_predictions(filename)
            # in case no outer folds exist, we write the inner_fold predictions
            else:
                return self.get_best_config_inner_fold_predictions(filename)

    def _get_best_outer_fold_configs_per_estimator(self) -> dict:
        # 1. find out which estimators there are
        last_element_name_identifier, last_element_dict = list(self.results.hyperpipe_info.elements.items())[-1]
        no_switch_found = False
        if not ":" in last_element_name_identifier:
            no_switch_found = True

        last_element_base_element, last_element_name = last_element_name_identifier.split(":")
        if not last_element_base_element == "SWITCH":
            no_switch_found = True

        if no_switch_found:
            logger.info("Could not identify switch at the end of the pipeline. Estimator Comparison aborted.")
            return

        # generate config key by switch name
        search_key = last_element_name + "__" + "estimator_name"
        estimator_list = last_element_dict.keys()
        best_configs_from_estimators = dict()
        for estimator in estimator_list:
            best_configs_from_estimators[estimator] = list()

        # 2. iterate list and filter configs
        for outer_fold in self.results.outer_folds:
            for estimator_name in estimator_list:
                try:
                    best_estimator_config = outer_fold.get_optimum_config(
                        metric=self.results.hyperpipe_info.best_config_metric,
                        maximize_metric=self.results.hyperpipe_info.
                        maximize_best_config_metric,
                        dict_filter=(search_key, estimator_name))
                    best_configs_from_estimators[estimator_name].append(best_estimator_config)
                except Warning as w:
                    logger.info("Could not find best config for estimator {} "
                                "in outer fold {}".format(estimator_name, outer_fold.fold_nr))

        return best_configs_from_estimators

    def get_n_best_validation_configs_per_estimator(self, n=10, estimator_names=None) -> dict:
        best_configs_from_estimator = self._get_best_outer_fold_configs_per_estimator()
        if estimator_names:
            all_estimators = list(best_configs_from_estimator.keys())
            for name in all_estimators:
                if name not in estimator_names:
                    del best_configs_from_estimator[name]

        best_n_config_dict = dict()
        for estimator_name, estimator_list in best_configs_from_estimator.items():
            if n > len(estimator_list):
                n_configs_per_estimator = [c.config_dict for c in estimator_list]
            else:
                sort_order = np.argsort([[c.get_test_metric(self.results.hyperpipe_info.best_config_metric, 'mean')
                                          for c in estimator_list]])
                n_configs_per_estimator = [estimator_list[idx].config_dict for idx in sort_order[:n]]
            best_n_config_dict[estimator_name] = n_configs_per_estimator
            print_config_list_table(estimator_name, n_configs_per_estimator)

        return best_n_config_dict

    def get_mean_of_best_validation_configs_per_estimator(self, write_to_file=False):

        best_configs_from_estimators = self._get_best_outer_fold_configs_per_estimator()

        # get mean values for each metric for each estimator config list
        estimator_performance_values = dict()
        for estimator_name, estimator_config_list in best_configs_from_estimators.items():
            estimator_performance_values[estimator_name] = dict()
            for metric in self.results.hyperpipe_info.metrics:
                performance_values = [c.get_test_metric(metric, 'mean')
                                      for c in estimator_config_list]
                estimator_performance_values[estimator_name][metric] = np.mean(performance_values)
        output = print_estimator_metrics(estimator_performance_values, self.results.hyperpipe_info.metrics, True)
        if write_to_file:
            text_file = open(os.path.join(self.output_settings.results_folder,
                                          "mean_best_estimator_performance.txt"), "w")
            text_file.write(output)
            text_file.close()
        return output

    def text_summary(self):
        def divider(header):
            return header.ljust(101, '=')

        output_string = divider("ANALYSIS INFORMATION ")

        elapsed_time = self.results.computation_end_time - self.results.computation_start_time
        output_string += """ 
Project Folder: {},
Computation Time: {} - {}
Duration: {}
Optimized for: {}
Hyperparameter Optimizer: {}

""".format(self.output_settings.results_folder,
           self.results.computation_start_time,
           self.results.computation_end_time,
           elapsed_time,
           self.results.hyperpipe_info.best_config_metric,
           self.results.hyperpipe_info.optimization["Optimizer"])

        output_string += divider("DUMMY RESULTS ")
        output_string += """
{}

""".format(print_metrics("DUMMY", self.results.dummy_estimator.get_test_metric(operation='mean'), summary=True))

        output_string += divider("AVERAGE PERFORMANCE ACROSS OUTER FOLDS ")

        test_metrics = self.results.get_test_metric_dict()
        train_metrics = self.results.get_train_metric_dict()
        output_string += """
{}

""".format(self.print_table_for_performance_overview(train_metrics, test_metrics))

        output_string += divider("BEST HYPERPARAMETER CONFIGURATION ")
        output_string += """
{}

""".format(json.dumps(self.results.best_config.human_readable_config, indent=4, sort_keys=True))

        output_string += """
{}

""".format(print_double_metrics(self.results.best_config.best_config_score.training.metrics,
                                self.results.best_config.best_config_score.validation.metrics,
                                summary=True))
        output_string += divider("PHOTONAI {} ".format(__version__))

        if self.output_settings.results_folder is not None:
            output_string += "\nYour results are stored in " + self.output_settings.results_folder + "\n"
            output_string += "Go to https://explorer.photon-ai.com and upload your photon_result_file.json " \
                             "for convenient result visualization! \n"
            output_string += "For more info and documentation visit https://www.photon-ai.com"

        if self.output_settings.save_output:
            try:
                summary_filename = os.path.join(self.output_settings.results_folder, 'photon_summary.txt')
                text_file = open(summary_filename, "w")
                text_file.write(output_string)
                text_file.close()
            except OSError as e:
                logger.error("Could not write summary file")
                logger.error(str(e))

            return output_string

    @staticmethod
    def print_table_for_performance_overview(metric_dict_train, metric_dict_test):
        x = PrettyTable()
        x.field_names = ["Metric Name", "Training Mean", "Training Std", "Test Mean", "Test Std"]
        for element_key, element_dict in metric_dict_train.items():
            x.add_row([element_key, element_dict["mean"], element_dict["std"],
                       metric_dict_test[element_key]["mean"], metric_dict_test[element_key]["std"]])
        return x
