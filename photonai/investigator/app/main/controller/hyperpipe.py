import numpy as np
from flask import render_template
from pymodm.errors import ValidationError, ConnectionError

from photonai import __version__
from photonai.processing.results_handler import ResultsHandler
from photonai.processing.results_structure import MDBHyperpipe
from .helper import load_pipe, load_available_pipes
from ..main import application
from ..model.BestConfigPlot import BestConfigPlot
from ..model.BestConfigTrace import BestConfigTrace
from ..model.Figures import plotly_optimizer_history, plot_scatter, plotly_confusion_matrix
from ..model.Metric import Metric
from ..model.PlotlyPlot import PlotlyPlot
from ..model.PlotlyTrace import PlotlyTrace


@application.route('/pipeline/<storage>')
def index_pipeline(storage):
    try:
        available_pipes = load_available_pipes()
        pipeline_list = list(MDBHyperpipe.objects.all())
        return render_template('pipeline/index.html', s=storage, pipelines=pipeline_list,
                               available_pipes=available_pipes)
    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message


@application.route('/error')
def show_error(msg):
    return render_template("default/error.html", error_msg=msg)


@application.route('/pipeline/<storage>/<name>')
def show_pipeline(storage, name):
    try:
        available_pipes = load_available_pipes()
        pipe = load_pipe(storage, name)
        # plot optimizer history
        handler = ResultsHandler(pipe)
        config_evaluations = handler.get_config_evaluations()
        min_config_evaluations = handler.get_minimum_config_evaluations()
        optimizer_history = plotly_optimizer_history('optimizer_history', config_evaluations, min_config_evaluations, pipe.hyperpipe_info.best_config_metric)
        plotly_dict = handler.eval_mean_time_components(write_results=False, plotly_return=True)

        # get information on cv, optimizer and data
        data_info = pipe.hyperpipe_info.data
        optimizer_info = pipe.hyperpipe_info.optimization
        cross_validation_info = pipe.hyperpipe_info.cross_validation

        # overall performance plots
        overview_plots = create_performance_overview_plots_new(pipe)

        # best config plots outer folds
        best_config_plot_list = create_best_config_plot(pipe)

        # confusion matrix or scatter plot
        predictions_plot_train, predictions_plot_test = create_prediction_plot(handler)

        return render_template('outer_folds/index.html',
                               pipe=pipe,
                               best_config_plot_list=best_config_plot_list,
                               overview_plots=overview_plots,
                               predictions_plot_train=predictions_plot_train,
                               predictions_plot_test=predictions_plot_test,
                               optimizer_history=optimizer_history,
                               s=storage,
                               available_pipes=available_pipes,
                               cross_validation_info=cross_validation_info,
                               data_info=data_info,
                               optimizer_info=optimizer_info,
                               time_monitor_pie=plotly_dict,
                               photon_version=__version__)
    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message


def create_performance_overview_plots_new(pipe):
    metrics = np.unique([metric.metric_name for metric in pipe.metrics_train])
    overview_plots = list()
    for metric in metrics:
        overview_plot = PlotlyPlot("overview_plot_" + metric, metric.replace("_", " "), show_legend=False,
                                   margin={'r': 30, 'l': 30})
        for fold in pipe.outer_folds:
            overview_plot_train_trace = PlotlyTrace("train_fold_" + str(fold.fold_nr), trace_color="train_color_bold",
                                                    trace_size=10)
            overview_plot_test_trace = PlotlyTrace("test_fold_" + str(fold.fold_nr), trace_color="alternative_test_color_bold",
                                                   trace_size=10)

            if fold.best_config:
                # save metrics for training dynamically in list
                overview_plot_train_trace.add_x('train')
                overview_plot_train_trace.add_y(fold.best_config.best_config_score.training.metrics[metric])
                overview_plot_test_trace.add_x('test')
                overview_plot_test_trace.add_y(fold.best_config.best_config_score.validation.metrics[metric])

            overview_plot.add_trace(overview_plot_train_trace)
            overview_plot.add_trace(overview_plot_test_trace)

        # add mean performance
        training_mean_trace = PlotlyTrace("mean_train", trace_size=4, trace_color="train_color", trace_type='bar',
                                          width=0.1)
        test_mean_trace = PlotlyTrace("mean_test", trace_size=4, trace_color="alternative_test_color", trace_type='bar',
                                      width=0.1)

        # add dummy performance
        # training_dummy_trace = PlotlyTrace("dummy_train", trace_size=4, trace_color="dummy_color", trace_type='bar',
        #                                   width=0.1, opacity=0, marker_line_width=3)
        test_dummy_trace = PlotlyTrace("dummy_test", trace_size=4, trace_color="dummy_color", trace_type='bar',
                                       width=0.1, opacity=0)
        #
        # for dummy_train in pipe.dummy_estimator.train:
        #     if dummy_train.metric_name == metric:
        #         if dummy_train.operation == 'FoldOperations.MEAN':
        #             training_dummy_trace.add_x('train')
        #             training_dummy_trace.add_y(dummy_train.value)

        for dummy_test in pipe.dummy_estimator.train:
            if dummy_test.metric_name == metric:
                if dummy_test.operation == 'FoldOperations.MEAN':
                    test_dummy_trace.add_x('dummy')
                    test_dummy_trace.add_y(dummy_test.value)

        for metric_train in pipe.metrics_train:
            if metric_train.metric_name == metric:
                if metric_train.operation == 'FoldOperations.MEAN':
                    training_mean_trace.add_x('train')
                    training_mean_trace.add_y(metric_train.value)

        for metric_test in pipe.metrics_test:
            if metric_test.metric_name == metric:
                if metric_test.operation == 'FoldOperations.MEAN':
                    test_mean_trace.add_x('test')
                    test_mean_trace.add_y(metric_test.value)

        overview_plot.add_trace(training_mean_trace)
        overview_plot.add_trace(test_mean_trace)
        # overview_plot.add_trace(training_dummy_trace)
        overview_plot.add_trace(test_dummy_trace)

        overview_plots.append(overview_plot)
    return overview_plots


def create_best_config_plot(pipe):
    best_config_plot_list = list()
    for fold in pipe.outer_folds:
        metric_training_list = list()
        metric_validation_list = list()
        if fold.best_config:
            # save metrics for training dynamically in list
            for key, value in fold.best_config.best_config_score.training.metrics.items():
                metric = Metric(key, value)
                metric_training_list.append(metric)

            # save metrics for validation dynamically in list
            for key, value in fold.best_config.best_config_score.validation.metrics.items():
                metric = Metric(key, value)
                metric_validation_list.append(metric)

        metric_training_trace = BestConfigTrace("training", metric_training_list, "", "bar")
        metric_test_trace = BestConfigTrace("test", metric_validation_list, "", "bar")

        best_config_plot = BestConfigPlot("outer_fold_" + str(fold.fold_nr) + "_best_config_overview",
                                          "Best Performance Outer Fold " + str(fold.fold_nr),
                                          metric_training_trace, metric_test_trace)
        best_config_plot_list.append(best_config_plot)
    return best_config_plot_list


def create_prediction_plot(handler):
    true_and_pred_val = list()
    true_and_pred_train = list()

    if handler.results.hyperpipe_info.eval_final_performance:
        for outer_fold in handler.results.outer_folds:
            outer_fold_values_val = handler.collect_fold_lists([outer_fold.best_config.best_config_score.validation],
                                                               [outer_fold.fold_nr])
            true_and_pred_val.append([outer_fold_values_val["y_true"], outer_fold_values_val["y_pred"]])

            outer_fold_values_train = handler.collect_fold_lists([outer_fold.best_config.best_config_score.training],
                                                               [outer_fold.fold_nr])
            true_and_pred_train.append([outer_fold_values_train["y_true"], outer_fold_values_train["y_pred"]])
    else:
        for inner_fold in handler.results.best_config.inner_folds:
            inner_fold_values_val = handler.collect_fold_lists([inner_fold.validation], [inner_fold.fold_nr])
            true_and_pred_val.append([inner_fold_values_val["y_true"], inner_fold_values_val["y_pred"]])
            inner_fold_values_train = handler.collect_fold_lists([inner_fold.training], [inner_fold.fold_nr])
            true_and_pred_train.append([inner_fold_values_train["y_true"], inner_fold_values_train["y_pred"]])

    predictions_plot_train = ""
    predictions_plot_test = ""
    if handler.results.hyperpipe_info.estimation_type == 'regressor':
        predictions_plot_train = plot_scatter(true_and_pred_train, 'predictions_plot_train',
                                              'True/Pred Training')
        predictions_plot_test = plot_scatter(true_and_pred_val, 'predictions_plot_test', 'True/Pred Test')
    else:
        predictions_plot_train = plotly_confusion_matrix('predictions_plot_train', 'Confusion Matrix Training',
                                                         true_and_pred_train)
        predictions_plot_test = plotly_confusion_matrix('predictions_plot_test', 'Confusion Matrix Test',
                                                        true_and_pred_val)
    return predictions_plot_train, predictions_plot_test
