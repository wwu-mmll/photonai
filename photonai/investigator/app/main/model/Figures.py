import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from photonai.processing.metrics import Scorer
from .PlotlyPlot import PlotlyPlot
from .PlotlyTrace import PlotlyTrace


def plotly_confusion_matrix(plot_name, title, folds):
    try:
        cms = list()
        y_true_all = list()
        y_pred_all = list()
        for y_true, y_pred in folds:
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

            cm = confusion_matrix(y_true, y_pred)
            # normalize
            cms.append(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        y_true_all = np.hstack(y_true_all)
        y_pred_all = np.hstack(y_pred_all)

        classes = list(unique_labels(y_true_all, y_pred_all))

        cms = np.asarray(cms)
        mean_cm = np.mean(cms, axis=0)
    except:
        return ''

    # build trace
    trace = PlotlyTrace('trace1')
    for single_class in classes:
        name = 'Class {}'.format(single_class + 1)
        trace.add_x(name)

    classes.reverse()
    for i, single_class in enumerate(classes):
        name = 'Class {}'.format(single_class + 1)
        trace.add_y(name)
        # todo: make class nr and index independent!!!!!!!
        trace.add_z(mean_cm[i, :])

    # create string from trace
    string_trace = "var trace1 = {type: 'heatmap'"
    string_trace += ", x: [" + trace.get_x_to_string() + "]"
    string_trace += ", y: [" + trace.get_y_to_string() + "]"
    string_trace += ", z: [" + trace.get_z_to_string(as_numbers=False) + "]"
    # string_trace += ", x: [" + str(trace.x) + "]"
    # string_trace += ", y: [" + str(trace.y) + "]"
    # string_trace += ", z: [" + str(trace.z) + "]"
    string_trace += """, zmin: '0',  zmax: '1',
    colorscale: [['0', 'rgb(255,245,240)'], ['0.2', 'rgb(254,224,210)'], ['0.4', 'rgb(252,187,161)'], ['0.5', 'rgb(252,146,114)'], ['0.6', 'rgb(251,106,74)'], ['0.7', 'rgb(239,59,44)'], ['0.8', 'rgb(203,24,29)'], ['0.9', 'rgb(165,15,21)'], ['1', 'rgb(103,0,13)']], 
    autocolorscale: false};"""


    plot = """
var data = [trace1];
var layout = {{
  title: '{}', 
  width: '400', 
  xaxis: {{
    title: 'Predicted Value', 
    titlefont: {{
      size: '18', 
      color: '7f7f7f'
    }}
  }}, 
  yaxis: {{
    title: 'True Value', 
    titlefont: {{
      size: '18', 
      color: '7f7f7f'
    }}
  }}, 
}};
Plotly.newPlot('{}', data, layout);""".format(title, plot_name)
    final_plot = string_trace + plot
    return final_plot


def plotly_optimizer_history(name, config_evaluations, minimum_config_evaluations, metric):

    # handle different lengths
    min_corresponding = len(min(config_evaluations[metric], key=len))
    config_evaluations_corres = [configs[:min_corresponding] for configs in config_evaluations[metric]]
    minimum_config_evaluations_corres = [configs[:min_corresponding] for configs in minimum_config_evaluations[metric]]

    mean = np.nanmean(np.asarray(config_evaluations_corres), axis=0)
    mean_min = np.nanmean(np.asarray(minimum_config_evaluations_corres), axis=0)

    greater_is_better = Scorer.greater_is_better_distinction(metric)
    if greater_is_better:
        caption = "Maximum"
    else:
        caption = "Minimum"

    # now do smoothing
    reduce_scatter_by = max([np.floor(min_corresponding / 75).astype(int), 1])

    traces = list()
    for i, fold in enumerate(config_evaluations[metric]):
        trace = PlotlyTrace("Fold_{}".format(i+1), trace_type='scatter', trace_size=6, trace_color="rgba(42, 54, 62, 0.5)")

        # add a few None so that list can be divided by smoothing_kernel
        remaining = len(fold) % reduce_scatter_by
        if remaining:
            fold.extend([np.nan] * (reduce_scatter_by - remaining))
        # calculate mean over every n elements so that plot is less cluttered
        reduced_fold = np.nanmean(np.asarray(fold).reshape(-1, reduce_scatter_by), axis=1)
        reduced_xfit = np.arange(max(1, reduce_scatter_by / 2), len(fold) + 1, step=reduce_scatter_by)

        trace.x = reduced_xfit
        trace.y = np.asarray(reduced_fold)
        traces.append(trace)

    trace = PlotlyTrace("Mean_{}_Performance".format(caption), trace_type='scatter', mode='lines', trace_size=8, trace_color="rgb(214, 123, 25)")
    trace.x = np.arange(1, len(mean_min) + 1)
    trace.y = mean_min
    traces.append(trace)

    for i, fold in enumerate(minimum_config_evaluations[metric]):
        trace = PlotlyTrace('Fold_{}_{}_Performance'.format(i+1, caption), trace_type='scatter', mode='lines',
                            trace_size=8, trace_color="rgba(214, 123, 25, 0.5)")
        xfit = np.arange(1, len(fold) + 1)
        trace.x = xfit
        trace.y = fold
        traces.append(trace)

    plot = PlotlyPlot(plot_name=name, title="Optimizer History", traces=traces, xlabel='No of Evaluations',
                      ylabel=metric.replace('_', ' '), show_legend=False)


    return plot.to_plot()


def plot_scatter(folds, name, title, colormap='nipy_spectral', trace_color=None):
    import seaborn as sns
    color = sns.hls_palette(len(folds), l=.3, s=.8)

    # cmap = getattr(plt.cm, colormap)
    # color = cmap(np.linspace(0, 1, len(folds)))

    list_final_value_validation_traces = list()
    fold_nr = 1
    for y_true, y_pred in folds:
        c = color[fold_nr-1]
        if trace_color:
            validation_trace = PlotlyTrace('Fold {}'.format(fold_nr), 'markers', 'scatter', trace_size=7,
                                           trace_color=trace_color)
        else:
            validation_trace = PlotlyTrace('Fold {}'.format(fold_nr), 'markers', 'scatter', trace_size=7,
                                       trace_color='rgba({}, {}, {}, 0.5)'.format(c[0], c[1], c[2]))
        for true_item in y_true:
            validation_trace.add_x(true_item)
        for pred_item in y_pred:
            validation_trace.add_y(pred_item)

        list_final_value_validation_traces.append(validation_trace)

        # create trace for regression line
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(np.reshape(y_true, (len(y_true), 1)), y_pred)
        regr_out = regr.predict(np.reshape(y_true, (len(y_true), 1)))
        r = pearsonr(y_true, y_pred)[0]
        if trace_color:
            regr_trace = PlotlyTrace('r={0:.2f}'.format(r), 'lines', 'scatter',
                                     trace_color=trace_color)
        else:
            regr_trace = PlotlyTrace('r={0:.2f}'.format(r), 'lines', 'scatter',
                                     trace_color='rgba({}, {}, {}, 0.8)'.format(c[0], c[1], c[2]))
        #regr_trace = PlotlyTrace('cool', 'lines', 'scatter', trace_color='black')
        for i, true_item in enumerate(y_true):
            regr_trace.add_x(true_item)
            regr_trace.add_y(regr_out[i])
        list_final_value_validation_traces.append(regr_trace)

        fold_nr += 1
    return PlotlyPlot(plot_name=name,
                     title=title,
                     traces=list_final_value_validation_traces,
                     show_legend=True,
                     xlabel='y_true',
                     ylabel='y_pred').to_plot()
