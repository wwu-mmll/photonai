from flask import render_template
from photonai.investigator.app.main import app
from photonai.validation.ResultsDatabase import MDBHyperpipe
from pymodm.errors import ValidationError, ConnectionError
from photonai.investigator.app.model.ConfigItem import ConfigItem
from photonai.investigator.app.model.Config import Config
from photonai.investigator.app.model.Metric import Metric
from photonai.investigator.app.model.PlotlyPlot import PlotlyPlot
from photonai.investigator.app.model.PlotlyTrace import PlotlyTrace
from photonai.investigator.app.model.BestConfigTrace import BestConfigTrace
from photonai.investigator.app.model.BestConfigPlot import BestConfigPlot


@app.route('/pipeline/<name>/outer_fold/<fold_nr>')
def show_outer_fold(name, fold_nr):
    try:
        # best config object always has just one fold
        default_fold_best_config = 0

        pipe = MDBHyperpipe.objects.get({'name': name})
        outer_fold = pipe.outer_folds[int(fold_nr) - 1]

        config_dict_list = list()
        error_plot_list = list()
        metric_training_list = list()
        metric_validation_list = list()

        # save metrics for training dynamically in list
        for key, value in outer_fold.best_config.inner_folds[default_fold_best_config].training.metrics.items():
            metric = Metric(key, value)
            metric_training_list.append(metric)

        # save metrics for validation dynamically in list
        for key, value in outer_fold.best_config.inner_folds[default_fold_best_config].validation.metrics.items():
            metric = Metric(key, value)
            metric_validation_list.append(metric)

        # building traces out of lists
        metric_training_trace = BestConfigTrace("training", metric_training_list, '', 'bar')
        metric_validation_trace = BestConfigTrace("validation", metric_validation_list, '', 'bar')

        # building plot out of traces
        best_config_plot = BestConfigPlot('best_config_overview', 'Best Configuration Overview', metric_training_trace, metric_validation_trace)

        # START building final values for training set (best config)
        count_item = int(0)
        true_training_trace = PlotlyTrace('y_true', 'markers', 'scatter')
        pred_training_trace = PlotlyTrace('y_pred', 'markers', 'scatter')

        for true_item in outer_fold.best_config.inner_folds[default_fold_best_config].training.y_true:
            true_training_trace.add_x(count_item)
            true_training_trace.add_y(true_item)
            count_item += 1

        # reset count variable
        count_item = int(0)

        for pred_item in outer_fold.best_config.inner_folds[default_fold_best_config].training.y_pred:
            pred_training_trace.add_x(count_item)
            pred_training_trace.add_y(pred_item)
            count_item += 1

        list_final_value_training_traces = list()
        list_final_value_training_traces.append(true_training_trace)
        list_final_value_training_traces.append(pred_training_trace)

        final_value_training_plot = PlotlyPlot('best_config_training_values', 'True/Predict for training set', list_final_value_training_traces)
        # END building final values for training set (best config)

        # START building final values for validation set (best config)
        count_item = int(0)
        true_validation_trace = PlotlyTrace('y_true', 'markers', 'scatter')
        pred_validation_trace = PlotlyTrace('y_pred', 'markers', 'scatter')

        for true_item in outer_fold.best_config.inner_folds[default_fold_best_config].validation.y_true:
            true_validation_trace.add_x(count_item)
            true_validation_trace.add_y(true_item)
            count_item += 1

        # reset count variable
        count_item = int(0)

        for pred_item in outer_fold.best_config.inner_folds[default_fold_best_config].validation.y_pred:
            pred_validation_trace.add_x(count_item)
            pred_validation_trace.add_y(pred_item)
            count_item += 1

        list_final_value_validation_traces = list()
        list_final_value_validation_traces.append(true_validation_trace)
        list_final_value_validation_traces.append(pred_validation_trace)

        final_value_validation_plot = PlotlyPlot('best_config_validation_values', 'True/Predict for validation set', list_final_value_validation_traces)
        # END building final values for validation set (best config)

        # START building plot objects for each tested config
        for config in outer_fold.tested_config_list:
            config_dict = Config('config_' + str(config.config_nr), config_nr=config.config_nr)

            error_plot_train = PlotlyPlot('train_config_' + str(config.config_nr), 'train error plot', [], show_legend=False)
            error_plot_test = PlotlyPlot('test_config_' + str(config.config_nr), 'test error plot', [], show_legend=False)

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

        return render_template('outer_folds/show.html', pipe=pipe, outer_fold=outer_fold
                               , error_plot_list=error_plot_list, bestConfigPlot=best_config_plot
                               , final_value_training_plot=final_value_training_plot
                               , final_value_validation_plot=final_value_validation_plot
                               , config_dict_list=config_dict_list)

    except ValidationError as exc:
        return exc.message
    except ConnectionError as exc:
        return exc.message
