from flask import render_template
from ..main import application
#from ..model.ResultsDatabase import MDBHyperpipe
from photonai.validation.ResultsDatabase import MDBHyperpipe
from photonai.validation.ResultsTreeHandler import ResultsTreeHandler
from pymodm.errors import ValidationError, ConnectionError
from ..model.Metric import Metric
from ..model.BestConfigTrace import BestConfigTrace
from ..model.BestConfigPlot import BestConfigPlot
from ..model.PlotlyTrace import PlotlyTrace
from ..model.PlotlyPlot import PlotlyPlot
from .helper import load_pipe, load_available_pipes
from ..model.Figures import plotly_optimizer_history, plot_scatter, plotly_confusion_matrix


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

        # if not isinstance(pipe, MDBHyperpipe):
        #     return render_template("default/error.html", error_msg=pipe)

        default_fold_best_config = 0

        # plot optimizer history
        handler = ResultsTreeHandler(pipe)
        config_evaluations = handler.get_config_evaluations()
        min_config_evaluations = handler.get_minimum_config_evaluations()
        optimizer_history = plotly_optimizer_history('optimizer_history', config_evaluations, min_config_evaluations, pipe.metrics[0])



        best_config_plot_list = list()

        # overview plot on top of the page
        overview_plot_train = PlotlyPlot("overview_plot_training", "Training Performance", show_legend=False)
        overview_plot_test = PlotlyPlot("overview_plot_test", "Test Performance", show_legend=False)

        # confusion matrix or scatter plot
        true_and_pred_val = list()
        true_and_pred_train = list()
        for fold in pipe.outer_folds:
            true_and_pred_val.append([fold.best_config.inner_folds[default_fold_best_config].validation.y_true,
                                  fold.best_config.inner_folds[default_fold_best_config].validation.y_pred])
            true_and_pred_train.append([fold.best_config.inner_folds[default_fold_best_config].training.y_true,
                                  fold.best_config.inner_folds[default_fold_best_config].training.y_pred])
        if pipe.estimation_type == 'regressor':
            predictions_plot_train = plot_scatter(true_and_pred_train, 'predictions_plot_train', 'True/Pred Training')
            predictions_plot_test = plot_scatter(true_and_pred_val, 'predictions_plot_test', 'True/Pred Test')
        else:
            predictions_plot_test = ''
            predictions_plot_train = ''

        for fold in pipe.outer_folds:

            overview_plot_training_trace = PlotlyTrace("fold_" + str(fold.fold_nr) + "_training", trace_color="rgb(91, 91, 91)")
            overview_plot_test_trace = PlotlyTrace("fold_" + str(fold.fold_nr) + "_test", trace_color="rgb(91, 91, 91)")

            if fold.best_config:

                metric_training_list = list()
                metric_validation_list = list()

                # save metrics for training dynamically in list
                for key, value in fold.best_config.inner_folds[default_fold_best_config].training.metrics.items():
                    overview_plot_training_trace.add_x(key)
                    overview_plot_training_trace.add_y(value)
                    metric = Metric(key, value)
                    metric_training_list.append(metric)

                # save metrics for validation dynamically in list
                for key, value in fold.best_config.inner_folds[default_fold_best_config].validation.metrics.items():
                    overview_plot_test_trace.add_x(key)
                    overview_plot_test_trace.add_y(value)
                    metric = Metric(key, value)
                    metric_validation_list.append(metric)

            overview_plot_train.add_trace(overview_plot_training_trace)
            overview_plot_test.add_trace(overview_plot_test_trace)

            metric_training_trace = BestConfigTrace("training", metric_training_list, "", "bar")
            metric_test_trace = BestConfigTrace("test", metric_validation_list, "", "bar")

            best_config_plot = BestConfigPlot("outer_fold_" + str(fold.fold_nr) + "_best_config_overview",
                                              "Best Performance Outer Fold " + str(fold.fold_nr),
                                              metric_training_trace, metric_test_trace)
            best_config_plot_list.append(best_config_plot)

        training_mean_trace = PlotlyTrace("mean", trace_size=8, trace_color="rgb(31, 119, 180)")
        test_mean_trace = PlotlyTrace("mean", trace_size=8, trace_color="rgb(214, 123, 25)")

        for metric in pipe.metrics_train:
            if metric.operation == 'FoldOperations.MEAN':
                training_mean_trace.add_x(metric.metric_name)
                training_mean_trace.add_y(metric.value)

        for metric in pipe.metrics_test:
            if metric.operation == 'FoldOperations.MEAN':
                test_mean_trace.add_x(metric.metric_name)
                test_mean_trace.add_y(metric.value)

        # # Start calculating mean values grouped by metrics and training or validation set
        # temp = {}
        # count = {}
        # for metric in metric_training_list:
        #     temp[metric.name] = 0
        #     count[metric.name] = 0
        #
        # for metric in metric_training_list:
        #     temp[metric.name] += float(metric.value)
        #     count[metric.name] += 1
        #
        # for key, value in temp.items():
        #     training_mean_trace.add_x(key)
        #     training_mean_trace.add_y(value / count[key])
        #
        # temp.clear()
        # count.clear()
        #
        # for metric in metric_validation_list:
        #     temp[metric.name] = 0
        #     count[metric.name] = 0
        #
        # for metric in metric_validation_list:
        #     temp[metric.name] += float(metric.value)
        #     count[metric.name] += 1
        #
        # for key, value in temp.items():
        #     test_mean_trace.add_x(key)
        #     test_mean_trace.add_y(value / count[key])
        # END calculating mean values grouped by metrics and training or validation set

        overview_plot_train.add_trace(training_mean_trace)
        overview_plot_test.add_trace(test_mean_trace)

        return render_template('outer_folds/index.html', pipe=pipe, best_config_plot_list=best_config_plot_list,
                               overview_plot_train=overview_plot_train,
                               overview_plot_test=overview_plot_test,
                               predictions_plot_train=predictions_plot_train,
                               predictions_plot_test=predictions_plot_test,
                               optimizer_history=optimizer_history,
                               s=storage,
                               available_pipes=available_pipes)
    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message
