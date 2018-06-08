from flask import render_template
from photonai.investigator.app.main import app
from photonai.validation.ResultsDatabase import MDBHyperpipe
from pymodm.errors import ValidationError, ConnectionError, DoesNotExist
from photonai.investigator.app.model.Metric import Metric
from photonai.investigator.app.model.BestConfigTrace import BestConfigTrace
from photonai.investigator.app.model.BestConfigPlot import BestConfigPlot
from photonai.investigator.app.model.PlotlyTrace import PlotlyTrace
from photonai.investigator.app.model.PlotlyPlot import PlotlyPlot
from photonai.investigator.app.controller.helper import load_pipe, load_available_pipes


@app.route('/pipeline/<storage>')
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


@app.route('/error')
def show_error(msg):
    return render_template("default/error.html", error_msg=msg)


@app.route('/pipeline/<storage>/<name>')
def show_pipeline(storage, name):

    try:

        available_pipes = load_available_pipes()
        pipe = load_pipe(storage, name)

        if not isinstance(pipe, MDBHyperpipe):
            return render_template("default/error.html", error_msg=pipe)

        default_fold_best_config = 0

        metric_training_list = list()
        metric_validation_list = list()
        best_config_plot_list = list()

        # overview plot on top of the page
        overview_plot_train = PlotlyPlot("overview_plot_training", "Training Performance", show_legend=False)
        overview_plot_test = PlotlyPlot("overview_plot_test", "Test Performance", show_legend=False)

        for fold in pipe.outer_folds:

            overview_plot_training_trace = PlotlyTrace("fold_" + str(fold.fold_nr) + "_training", trace_color="rgb(91, 91, 91)")
            overview_plot_test_trace = PlotlyTrace("fold_" + str(fold.fold_nr) + "_test", trace_color="rgb(91, 91, 91)")

            if fold.best_config:

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

        # Start calculating mean values grouped by metrics and training or validation set
        temp = {}
        count = {}

        for metric in metric_training_list:
            temp[metric.name] = 0
            count[metric.name] = 0

        for metric in metric_training_list:
            temp[metric.name] += float(metric.value)
            count[metric.name] += 1

        for key, value in temp.items():
            training_mean_trace.add_x(key)
            training_mean_trace.add_y(value / count[key])

        temp.clear()
        count.clear()

        for metric in metric_validation_list:
            temp[metric.name] = 0
            count[metric.name] = 0

        for metric in metric_validation_list:
            temp[metric.name] += float(metric.value)
            count[metric.name] += 1

        for key, value in temp.items():
            test_mean_trace.add_x(key)
            test_mean_trace.add_y(value / count[key])
        # END calculating mean values grouped by metrics and training or validation set

        overview_plot_train.add_trace(training_mean_trace)
        overview_plot_test.add_trace(test_mean_trace)

        return render_template('outer_folds/index.html', pipe=pipe, best_config_plot_list=best_config_plot_list,
                               overview_plot_train=overview_plot_train,
                               overview_plot_test=overview_plot_test,
                               s=storage,
                               available_pipes=available_pipes)
    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message
