import itertools
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import sem
from sklearn.metrics import confusion_matrix, roc_curve
from pymodm import connect

from ..validation.ResultsDatabase import MDBHyperpipe
from ..photonlogger.Logger import Logger
#from ..base.PhotonBase.Hyp import OptimizerMetric


class ResultsTreeHandler:
    def __init__(self, result_tree: MDBHyperpipe = None):
        self.results = result_tree

    def load_from_file(self, results_file: str):
        self.results = MDBHyperpipe.from_document(pickle.load(open(results_file, 'rb')))

    def load_from_mongodb(self, mongodb_connect_url: str, pipe_name: str):
        connect(mongodb_connect_url)
        self.results = list(MDBHyperpipe.objects.raw({'_id': pipe_name}))[0]


    @staticmethod
    def get_methods():
        """
        This function returns a list of all methods available for ResultsTreeHandler.
        """
        methods_list = [s for s in dir(ResultsTreeHandler) if not '__' in s]
        return methods_list

    def get_performance_table(self):
        """
        This function returns a summary table of the overall results.
        ToDo: add best_config information!
        """

        res_tab = pd.DataFrame()
        for i, folds in enumerate(self.results.outer_folds):
            # add best config infos
            try:
                res_tab.loc[i, 'best_config'] = folds.best_config.human_readable_config
            except:
                res_tab.loc[i, 'best_config'] = str(folds.best_config.human_readable_config)

            # add fold index
            res_tab.loc[i, 'fold'] = folds.fold_nr

            # add sample size infos
            res_tab.loc[i, 'n_train'] = folds.best_config.inner_folds[0].number_samples_training
            res_tab.loc[i, 'n_validation'] = folds.best_config.inner_folds[0].number_samples_validation

            # add performance metrics
            d = folds.best_config.inner_folds[0].validation.metrics
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
        for metric in self.results.outer_folds[0].best_config.inner_folds[0].validation.metrics.keys():
            performances[metric] = list()
        for i, fold in enumerate(self.results.outer_folds):
            for metric, value in fold.best_config.inner_folds[0].validation.metrics.items():
                performances[metric].append(value)
        return performances

    def get_config_evaluations(self):
        """
        Return the test performance of every tested configuration in every outer fold.
        :return:
        """
        config_performances = list()
        for outer_fold in self.results.outer_folds:
            performance = dict()
            for metric in self.results.metrics:
                performance[metric] = list()

            for config in outer_fold.tested_config_list:
                for metric in self.results.metrics:
                    if config.config_failed:
                        performance[metric].append(np.nan)
                    else:
                        for item in config.metrics_test:
                            if (item.operation == 'FoldOperations.MEAN') and (item.metric_name == metric):
                                performance[metric].append(item.value)
            config_performances.append(performance)

        config_performances_dict = dict()
        for metric in self.results.metrics:
            config_performances_dict[metric] = list()
            for fold in config_performances:
                config_performances_dict[metric].append(fold[metric])

        return config_performances_dict

    def get_minimum_config_evaluations(self):
        config_evaluations = self.get_config_evaluations()
        minimum_config_evaluations = dict()

        for metric, evaluations in config_evaluations.items():
            minimum_config_evaluations[metric] = list()
            greater_is_better = OptimizerMetric.greater_is_better_distinction(metric)

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

    def plot_optimizer_history(self, metric,
                               title: str = 'Optimizer History',
                               type: str = 'plot',
                               smoothing_kernel: int = 1,
                               file: str = None):
        """

        :param metric: specify metric that has been stored within the PHOTON results tree
        :param type: 'plot' or 'scatter'
        :param smoothing_kernel: integer, takes running average over array to smooth visualization
        :param file: specify a filename if you want to save the plot
        :return:
        """
        def smooth_array(x, N):
            return np.convolve(x, np.ones((N,)) / N, mode='same')

        if metric not in self.results.metrics:
            raise ValueError('Metric "{}" not stored in results tree'.format(metric))

        config_evaluations = self.get_config_evaluations()
        minimum_config_evaluations = self.get_minimum_config_evaluations()

        mean = np.nanmean(np.asarray(config_evaluations[metric]), axis=0)
        std = np.nanstd(np.asarray(config_evaluations[metric]), axis=0)
        mean_min = np.nanmean(np.asarray(minimum_config_evaluations[metric]), axis=0)
        std_min = np.nanstd(np.asarray(minimum_config_evaluations[metric]), axis=0)

        # now do smoothing
        if smoothing_kernel > 1:
            mean = smooth_array(mean, smoothing_kernel)
            std = smooth_array(std, smoothing_kernel)
            mean_min = smooth_array(mean_min, smoothing_kernel)
            std_min = smooth_array(std_min, smoothing_kernel)

        xfit = np.arange(0, len(mean_min))
        yfit = mean_min
        dyfit = std_min

        plt.plot(xfit, yfit, '-', color='gray', label='Minimum Performance of Configs (MEAN)')
        plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2, label='Minimum Performance of Configs (STD)')

        if type == 'plot':
            plt.plot(xfit, mean, '-', color='red', label='Performance of Configs (MEAN)')
            plt.fill_between(xfit, mean - std, mean + std, color='red', alpha=0.2, label='Performance of Configs (STD)')
        elif type == 'scatter':
            xfit = np.tile(xfit, len(config_evaluations[metric]))[0::smoothing_kernel]
            folds = list()
            for fold in config_evaluations[metric]:
                folds.append(np.mean(np.asarray(fold).reshape(-1, smoothing_kernel), axis=1))
            yfit = np.asarray(folds).flatten()
            # calculate mean over n configuration --> this makes the plot less cluttered; n = smoothing_kernel
            plt.scatter(xfit, yfit, color='gray', alpha=0.3, label='Performance of Configs', marker='.')
        else:
            raise ValueError('Please specify either "plot" or "scatter".')

        plt.ylabel(metric.replace('_', ' '))
        plt.xlabel('No of Evaluations')
        plt.legend()
        plt.title(title)
        if file:
            plt.savefig(file)
        plt.show()

    def get_val_preds(self, sort_CV=True):
        """
        This function returns the predictions, true targets, and fold index
        for the best configuration of each outer fold.
        """
        y_true = []
        y_pred = []
        sample_inds = []
        y_pred_probabilities = []
        fold_idx = []
        for i, fold in enumerate(self.results.outer_folds):
            n_samples = len(fold.best_config.inner_folds[0].validation.y_true)
            y_true.extend(fold.best_config.inner_folds[0].validation.y_true)
            y_pred.extend(fold.best_config.inner_folds[0].validation.y_pred)
            y_pred_probabilities.extend(fold.best_config.inner_folds[0].validation.probabilities)
            fold_idx.extend(np.repeat(i, n_samples))
            if sort_CV:
                sample_inds.extend(fold['validation']['indices'])
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred_probabilities = np.asarray(y_pred_probabilities)
        fold_idx = np.asarray(fold_idx)
        if sort_CV:
            sample_inds = np.asarray(sample_inds)
            sort_index = np.argsort(sample_inds)
            y_true = y_true[sort_index]
            y_pred = y_pred[sort_index]
            if len(y_pred_probabilities) != 0:
                y_pred_probabilities = y_pred_probabilities[sort_index]

        return {'y_true': y_true, 'y_pred': y_pred, 'sample_inds_CV': sample_inds,
                'y_pred_probabilities': y_pred_probabilities, 'fold_indices': fold_idx}

    def get_inner_val_preds(self, sort_CV=True, config_no=0):
        """
        This function returns the predictions, true targets, and fold index
        for the best configuration of each inner fold if outer fold is not set and eval_final_performance is False
        AND there is only 1 config tested!
        :param sort_CV: sort predictions to match input sequence (i.e. undo CV shuffle = True)?
        :param config_no: which tested config to use?
        """
        y_true = []
        y_pred = []
        if sort_CV:
            sample_inds = []
        y_pred_probabilities = []
        fold_idx = []
        for i, fold in enumerate(self.results._data['outer_folds'][0]['tested_config_list'][config_no]['inner_folds']):
            n_samples = len(fold['validation']['y_true'])
            y_true.extend(fold['validation']['y_true'])
            y_pred.extend(fold['validation']['y_pred'])
            y_pred_probabilities.extend(fold['validation']['probabilities'])
            fold_idx.extend(np.repeat(i, n_samples))
            if sort_CV:
                sample_inds.extend(fold['validation']['indices'])
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred_probabilities = np.asarray(y_pred_probabilities)
        fold_idx = np.asarray(fold_idx)
        if sort_CV:
            sample_inds = np.asarray(sample_inds)
            sort_index = np.argsort(sample_inds)
            y_true = y_true[sort_index]
            y_pred = y_pred[sort_index]
            if len(y_pred_probabilities) != 0:
                y_pred_probabilities = y_pred_probabilities[sort_index]

        return {'y_true': y_true, 'y_pred': y_pred,
                'y_pred_probabilities': y_pred_probabilities, 'fold_indices': fold_idx}

    def get_importance_scores(self):
        """
        This function returns the importance scores for the best configuration of each outer fold.
        """
        imps = []
        for i, fold in enumerate(self.results.outer_folds):
            imps.append(fold.best_config.inner_folds[0].training.feature_importances)
        return imps

    def plot_true_pred(self, confidence_interval=95):
        """
        This function plots predictions vs. (true) targets and plots a regression line
        with confidence interval.
        """
        preds = ResultsTreeHandler.get_val_preds(self)
        ax = sns.regplot(x=preds['y_pred'], y=preds['y_true'], ci=confidence_interval)
        ax.set(xlabel='Predicted Values', ylabel='True Values')
        plt.show()

    def plot_confusion_matrix(self, classes=None, normalize=False, title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        preds = ResultsTreeHandler.get_val_preds(self)
        cm = confusion_matrix(preds['y_true'], preds['y_pred'])
        np.set_printoptions(precision=2)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            Logger().info("Normalized confusion matrix")
        else:
            Logger().info('Confusion matrix')
        Logger().info(cm)

        plt.figure()
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if classes == None:
            classes = ['class ' + str(c + 1) for c in np.unique(preds['y_true'])]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plotlyFig = ResultsTreeHandler.__plotlyfy(plt)
        plt.show()

    def plot_roc_curve(self, pos_label=1, y_score_col=1):
        """
        This function plots the ROC curve.
        :param pos_label: In binary classiciation, what is the positive class label?
        :param y_score_col: In binary classiciation, which column of the probability matrix contains the positive class probabilities?
        :return: None
        """


        # get predictive probabilities
        preds = ResultsTreeHandler.get_val_preds(self)

        # get ROC infos
        fpr, tpr, _ = roc_curve(y_true=preds['y_true'],
                                y_score=preds['y_pred_probabilities'][:, y_score_col],
                                pos_label=pos_label)

        # plot ROC curve
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='best')
        plt.show()

