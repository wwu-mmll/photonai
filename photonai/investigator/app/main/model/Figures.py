from sklearn.metrics import confusion_matrix
from photonai.base.PhotonBase import Hyperpipe
import numpy as np
from .PlotlyTrace import PlotlyTrace
from .PlotlyPlot import PlotlyPlot


def plotly_confusion_matrix(name, y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
    except:
        return ''

    trace = """var trace1 = {{
    type: 'heatmap', 
    x: ['Class 1', 'Class 2'], 
    y: ['Class 2', 'Class 1'], 
    zmin: '1', 
    z: [['{}', '{}'], ['{}', '{}']], 
    colorscale: [['0', 'rgb(255,245,240)'], ['0.2', 'rgb(254,224,210)'], ['0.4', 'rgb(252,187,161)'], ['0.5', 'rgb(252,146,114)'], ['0.6', 'rgb(251,106,74)'], ['0.7', 'rgb(239,59,44)'], ['0.8', 'rgb(203,24,29)'], ['0.9', 'rgb(165,15,21)'], ['1', 'rgb(103,0,13)']], 
    autocolorscale: false}};""".format(cm[1, 0], cm[1, 1], cm[0, 0], cm[0, 1])

    plot = """
var data = [trace1];
var layout = {{
  title: 'Confusion Matrix', 
  width: '400', 
  xaxis: {{
    title: 'Predicted value', 
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
Plotly.newPlot('{}', data, layout);""".format(name)
    final_plot = trace + plot
    return final_plot


def plotly_optimizer_history(name, config_evaluations, minimum_config_evaluations, metric):

    # handle different lengths
    min_corresponding = len(min(config_evaluations[metric], key=len))
    config_evaluations_corres = [configs[:min_corresponding] for configs in config_evaluations[metric]]
    minimum_config_evaluations_corres = [configs[:min_corresponding] for configs in minimum_config_evaluations[metric]]

    mean = np.nanmean(np.asarray(config_evaluations_corres), axis=0)
    mean_min = np.nanmean(np.asarray(minimum_config_evaluations_corres), axis=0)

    greater_is_better = Hyperpipe.Optimization.greater_is_better_distinction(metric)
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
        reduced_xfit = np.arange(reduce_scatter_by / 2, len(fold), step=reduce_scatter_by)

        trace.x = reduced_xfit
        trace.y = np.asarray(reduced_fold)
        traces.append(trace)

    trace = PlotlyTrace("Mean_{}_Performance".format(caption), trace_type='scatter', mode='lines', trace_size=8, trace_color="rgb(214, 123, 25)")
    trace.x = np.arange(0, len(mean_min))
    trace.y = mean_min
    traces.append(trace)

    for i, fold in enumerate(minimum_config_evaluations[metric]):
        trace = PlotlyTrace('Fold_{}_{}_Performance'.format(i+1, caption), trace_type='scatter', mode='lines',
                            trace_size=8, trace_color="rgba(214, 123, 25, 0.5)")
        xfit = np.arange(0, len(fold))
        trace.x = xfit
        trace.y = fold
        traces.append(trace)

    plot = PlotlyPlot(plot_name=name, title="Optimizer History", traces=traces, xlabel='No of Evaluations',
                      ylabel=metric.replace('_', ' '), show_legend=False)

    #plot = PlotlyPlot(plot_name=name, title="Optimizer History", traces=traces)

    return plot.to_plot()
