from flask import render_template
from ..main import application
from pymodm.errors import ValidationError, ConnectionError
from ..model.ConfigItem import ConfigItem
from ..model.Config import Config
from ..model.Metric import Metric
from ..model.PlotlyPlot import PlotlyPlot
from ..model.PlotlyTrace import PlotlyTrace
from ..model.BestConfigTrace import BestConfigTrace
from ..model.BestConfigPlot import BestConfigPlot
from ..model.Figures import plotly_confusion_matrix, plot_scatter
from .helper import load_pipe, load_available_pipes


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

        # building traces out of lists
        metric_training_trace = BestConfigTrace("training", metric_training_list, '', 'bar')
        metric_validation_trace = BestConfigTrace("validation", metric_validation_list, '', 'bar')

        # building plot out of traces
        best_config_plot = BestConfigPlot('best_config_overview', 'Best Configuration Overview', metric_training_trace, metric_validation_trace)

        #---------------------
        # Training
        #---------------------
        # START building final values for training set (best config)
        y_true = outer_fold.best_config.best_config_score.training.y_true
        y_pred = outer_fold.best_config.best_config_score.training.y_pred
        if pipe.estimation_type == 'classifier':
            final_value_training_plot = plotly_confusion_matrix('best_config_training_values', 'Confusion Matrix Train',
                                        [[y_true, y_pred]])
        else:
            final_value_training_plot = plot_scatter([[y_true, y_pred]],
                                                     title='True/Predict for Training Set',
                                                     name='best_config_training_values')
        # END building final values for training set (best config)

        #---------------------
        # Validation
        #---------------------
        y_true = outer_fold.best_config.best_config_score.validation.y_true
        y_pred = outer_fold.best_config.best_config_score.validation.y_pred
        if pipe.estimation_type == 'classifier':
            final_value_validation_plot = plotly_confusion_matrix('best_config_validation_values', 'Confusion Matrix Test',
                                        [[y_true, y_pred]])
        else:
            # START building final values for validation set (best config)
            final_value_validation_plot = plot_scatter([[y_true, y_pred]],
                                                       title='True/Predict for Validation Set',
                                                       name='best_config_validation_values')
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

            for key, value in config.config_dict.items():
                config_item = ConfigItem(str(key), str(value))
                config_dict.add_item(config_item)
            config_dict_list.append(config_dict)

            error_plot_train.add_trace(trace_training)
            error_plot_test.add_trace(trace_test)

            error_plot_list.append(error_plot_train)
            error_plot_list.append(error_plot_test)
        # END building plot objects for each tested config

        return render_template('outer_folds/show.html', pipe=pipe, outer_fold=outer_fold,
                               error_plot_list=error_plot_list, bestConfigPlot=best_config_plot,
                               final_value_training_plot=final_value_training_plot,
                               final_value_validation_plot=final_value_validation_plot,
                               config_dict_list=config_dict_list,
                               s=storage,
                               available_pipes=available_pipes)

    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message
