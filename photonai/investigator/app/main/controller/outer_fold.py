from flask import render_template
from pymodm.errors import ValidationError, ConnectionError

from .helper import load_pipe, load_available_pipes
from ..main import application
from ..model.Config import Config
from ..model.ConfigItem import ConfigItem
from ..model.Figures import plotly_confusion_matrix, plot_scatter
from ..model.Metric import Metric
from ..model.PlotlyPlot import PlotlyPlot
from ..model.PlotlyTrace import PlotlyTrace


@application.route('/pipeline/<storage>/<name>/outer_fold/<fold_nr>')
def show_outer_fold(storage, name, fold_nr):
    try:
        available_pipes = load_available_pipes()
        pipe = load_pipe(storage, name)

        outer_fold = pipe.outer_folds[int(fold_nr) - 1]
        config_dict_list = list()
        error_plot_list = list()
        metric_training_list = list()
        metric_validation_list = list()

        # save metrics for training dynamically in list
        for key, value in outer_fold.best_config.best_config_score.training.metrics.items():
            metric = Metric(key, value)
            metric_training_list.append(metric)

        # save metrics for validation dynamically in list
        for key, value in outer_fold.best_config.best_config_score.validation.metrics.items():
            metric = Metric(key, value)
            metric_validation_list.append(metric)

        best_config_plots = list()
        for metric in outer_fold.best_config.best_config_score.validation.metrics.keys():
            best_config_plot = PlotlyPlot('fold_performance_' + metric, metric.replace("_", " "), show_legend=False,
                                          margin={'r': 30, 'l': 30})
            # add mean performance
            training_mean_trace = PlotlyTrace("mean_train", trace_size=4, trace_color="train_color", trace_type='bar',
                                              width=0.1)
            test_mean_trace = PlotlyTrace("mean_test", trace_size=4, trace_color="alternative_test_color", trace_type='bar',
                                          width=0.1)
            for metric_train in metric_training_list:
                if metric_train.name == metric:
                    training_mean_trace.add_x('train')
                    training_mean_trace.add_y(metric_train.value)

            for metric_test in metric_validation_list:
                if metric_test.name == metric:
                    test_mean_trace.add_x('test')
                    test_mean_trace.add_y(metric_test.value)
            best_config_plot.add_trace(training_mean_trace)
            best_config_plot.add_trace(test_mean_trace)
            best_config_plots.append(best_config_plot)

        #---------------------
        # Training
        #---------------------
        if not pipe.hyperpipe_info.eval_final_performance:
            final_value_training_plot = ""
            final_value_validation_plot = ""
        else:
            # START building final values for training set (best config)
            y_true = outer_fold.best_config.best_config_score.training.y_true
            y_pred = outer_fold.best_config.best_config_score.training.y_pred
            if pipe.hyperpipe_info.estimation_type == 'classifier':
                final_value_training_plot = plotly_confusion_matrix('best_config_training_values',
                                                                    'Confusion Matrix Train',
                                                                    [[y_true, y_pred]])
            else:
                final_value_training_plot = plot_scatter([[y_true, y_pred]],
                                                         title='True/Predict for Training Set',
                                                         name='best_config_training_values', trace_color='train_color')
            # END building final values for training set (best config)

            # ---------------------
            # Validation
            # ---------------------
            y_true = outer_fold.best_config.best_config_score.validation.y_true
            y_pred = outer_fold.best_config.best_config_score.validation.y_pred
            if pipe.hyperpipe_info.estimation_type == 'classifier':
                final_value_validation_plot = plotly_confusion_matrix('best_config_validation_values',
                                                                      'Confusion Matrix Test',
                                                                      [[y_true, y_pred]])
            else:
                # START building final values for validation set (best config)
                final_value_validation_plot = plot_scatter([[y_true, y_pred]],
                                                           title='True/Predict for Test Set',
                                                           name='best_config_validation_values',
                                                           trace_color='alternative_test_color')
                # END building final values for validation set (best config)

        # START building plot objects for each tested config
        for config in outer_fold.tested_config_list:
            config_dict = Config('config_' + str(config.config_nr), config_nr=config.config_nr)

            error_plot_train = PlotlyPlot('train_config_' + str(config.config_nr), 'Inner Training Performance', [], show_legend=False)
            error_plot_test = PlotlyPlot('test_config_' + str(config.config_nr), 'Validation Performance', [], show_legend=False)

            for inner_fold in config.inner_folds:
                trace_training_metrics = PlotlyTrace('training_fold_' + str(inner_fold.fold_nr), 'markers', 'scatter', trace_color="rgb(91, 91, 91)")
                trace_test_metrics = PlotlyTrace('test_fold_' + str(inner_fold.fold_nr), 'markers', 'scatter', trace_color="rgb(91, 91, 91)")
                for key, value in inner_fold.training.metrics.items():
                    trace_training_metrics.add_x(key)
                    trace_training_metrics.add_y(value)
                for key, value in inner_fold.validation.metrics.items():
                    trace_test_metrics.add_x(key)
                    trace_test_metrics.add_y(value)

                error_plot_train.add_trace(trace_training_metrics)
                error_plot_test.add_trace(trace_test_metrics)

            trace_training = PlotlyTrace('training_mean_' + str(config.config_nr), 'markers', 'scatter', trace_size=8, with_error=True)
            trace_test = PlotlyTrace('test_mean_' + str(config.config_nr), 'markers', 'scatter', trace_size=8, with_error=True)

            for train in config.metrics_train:
                if train.operation == 'FoldOperations.MEAN':
                    trace_training.add_x(str(train.metric_name))
                    trace_training.add_y(train.value)
                elif train.operation == 'FoldOperations.STD':
                    trace_training.add_error_y(train.value)
            for test in config.metrics_test:
                if test.operation == 'FoldOperations.MEAN':
                    trace_test.add_x(str(test.metric_name))
                    trace_test.add_y(test.value)
                elif test.operation == 'FoldOperations.STD':
                    trace_test.add_error_y(test.value)

            for key, value in config.human_readable_config.items():
                config_item = ConfigItem(str(key), str(value))
                config_dict.add_item(config_item)
            config_dict_list.append(config_dict)

            error_plot_train.add_trace(trace_training)
            error_plot_test.add_trace(trace_test)

            error_plot_list.append(error_plot_train)
            error_plot_list.append(error_plot_test)
        # END building plot objects for each tested config

        return render_template('outer_folds/show.html', pipe=pipe, outer_fold=outer_fold,
                               error_plot_list=error_plot_list, best_config_plots=best_config_plots,
                               final_value_training_plot=final_value_training_plot,
                               final_value_validation_plot=final_value_validation_plot,
                               config_dict_list=config_dict_list,
                               s=storage,
                               available_pipes=available_pipes)

    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message
